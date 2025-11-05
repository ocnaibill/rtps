"""
Ray Tracing em TEMPO REAL com GPU - MacBook M4
Demonstração interativa de reflexões e iluminação física

Instalar: pip install taichi numpy

Controles:
- W/S: Move câmera para frente/trás
- A/D: Move câmera para esquerda/direita  
- Q/E: Move câmera para cima/baixo
- Mouse: Rotaciona câmera (clique e arraste)
- ESC: Sair
"""

import taichi as ti
import numpy as np
import math

# Inicializa Taichi com backend Metal (GPU do Macbook M4)
ti.init(arch=ti.metal)

# Parâmetros da imagem (reduzido para performance em tempo real)
WIDTH, HEIGHT = 1920, 1080
ASPECT_RATIO = WIDTH / HEIGHT

# Buffers de imagem na GPU
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# Variáveis de câmera
camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
camera_angle = ti.field(dtype=ti.f32, shape=(2,))  # yaw, pitch
time_var = ti.field(dtype=ti.f32, shape=())

# === CONCEITOS FÍSICOS ===

@ti.func
def reflect(incident: ti.math.vec3, normal: ti.math.vec3) -> ti.math.vec3:
    """
    Lei da Reflexão: θ_i = θ_r
    Vetor refletido: R = I - 2(I·N)N
    """
    return incident - 2.0 * ti.math.dot(incident, normal) * normal

@ti.func
def fresnel_schlick(cos_theta: ti.f32, F0: ti.f32) -> ti.f32:
    """
    Aproximação de Schlick para equações de Fresnel
    F(θ) = F0 + (1 - F0)(1 - cos θ)^5
    """
    return F0 + (1.0 - F0) * ti.pow(1.0 - cos_theta, 5.0)

@ti.dataclass
class Ray:
    origin: ti.math.vec3
    direction: ti.math.vec3

@ti.dataclass
class Sphere:
    center: ti.math.vec3
    radius: ti.f32
    color: ti.math.vec3
    reflectivity: ti.f32
    F0: ti.f32

@ti.dataclass
class HitRecord:
    hit: ti.i32
    t: ti.f32
    point: ti.math.vec3
    normal: ti.math.vec3
    color: ti.math.vec3
    reflectivity: ti.f32
    F0: ti.f32

# Define a cena (esferas) no escopo GLOBAL
num_spheres = 5
spheres = Sphere.field(shape=(num_spheres,))

@ti.func
def intersect_sphere(ray: Ray, sphere: Sphere) -> HitRecord:
    """
    Interseção raio-esfera
    Equação quadrática: t² + 2t(D·(O-C)) + ||O-C||² - r² = 0
    """
    record = HitRecord(hit=0, t=1e10, point=ti.math.vec3(0.0), 
                      normal=ti.math.vec3(0.0), color=ti.math.vec3(0.0),
                      reflectivity=0.0, F0=0.0)
    
    oc = ray.origin - sphere.center
    a = ti.math.dot(ray.direction, ray.direction)
    b = 2.0 * ti.math.dot(oc, ray.direction)
    c = ti.math.dot(oc, oc) - sphere.radius * sphere.radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant > 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        if t > 0.001:
            record.hit = 1
            record.t = t
            record.point = ray.origin + t * ray.direction
            record.normal = ti.math.normalize(record.point - sphere.center)
            record.color = sphere.color
            record.reflectivity = sphere.reflectivity
            record.F0 = sphere.F0
    
    return record

@ti.func
def trace_ray(ray: Ray, spheres: ti.template(), num_spheres: ti.i32, 
              light_pos: ti.math.vec3, depth: ti.i32) -> ti.math.vec3:
    """
    Traça raio com reflexões físicas
    Modelo de Phong + Fresnel
    """
    color = ti.math.vec3(0.0, 0.0, 0.0)
    attenuation = ti.math.vec3(1.0, 1.0, 1.0)
    current_ray = ray
    
    for bounce in range(depth):
        # Encontra intersecção mais próxima
        closest = HitRecord(hit=0, t=1e10, point=ti.math.vec3(0.0),
                           normal=ti.math.vec3(0.0), color=ti.math.vec3(0.0),
                           reflectivity=0.0, F0=0.0)
        
        for i in range(num_spheres):
            hit = intersect_sphere(current_ray, spheres[i])
            if hit.hit == 1 and hit.t < closest.t:
                closest = hit
        
        if closest.hit == 0:
            # Gradiente de céu animado
            t = 0.5 * (current_ray.direction.y + 1.0)
            sky = (1.0 - t) * ti.math.vec3(1.0, 1.0, 1.0) + t * ti.math.vec3(0.5, 0.7, 1.0)
            color += attenuation * sky
            break
        
        # === MODELO DE ILUMINAÇÃO DE PHONG ===
        
        ambient = 0.1 * closest.color
        
        # Difusa (Lei de Lambert)
        light_dir = ti.math.normalize(light_pos - closest.point)
        diff = ti.max(ti.math.dot(closest.normal, light_dir), 0.0)
        diffuse = diff * closest.color
        
        # Especular
        view_dir = ti.math.normalize(-current_ray.direction)
        reflect_dir = reflect(-light_dir, closest.normal)
        spec = ti.pow(ti.max(ti.math.dot(view_dir, reflect_dir), 0.0), 32.0)
        specular = 0.5 * spec * ti.math.vec3(1.0, 1.0, 1.0)
        
        local_color = ambient + diffuse + specular
        
        # Fresnel
        cos_theta = ti.max(ti.math.dot(-current_ray.direction, closest.normal), 0.0)
        fresnel = fresnel_schlick(cos_theta, closest.F0)
        reflectance = closest.reflectivity * fresnel
        
        color += attenuation * (1.0 - reflectance) * local_color
        
        if reflectance < 0.01:
            break
            
        attenuation *= reflectance
        current_ray.origin = closest.point
        current_ray.direction = reflect(current_ray.direction, closest.normal)
    
    return color

@ti.func
def get_camera_ray(i: ti.i32, j: ti.i32, cam_pos: ti.math.vec3, 
                   yaw: ti.f32, pitch: ti.f32) -> Ray:
    """
    Cria raio da câmera com rotação
    """
    # Coordenadas normalizadas
    u = (2.0 * i / WIDTH - 1.0) * ASPECT_RATIO
    # Inverte v para corrigir imagem "de cabeça pra baixo"
    v = 2.0 * j / HEIGHT - 1.0
    
    # Direção base (olhando para -Z)
    local_dir = ti.math.vec3(u, v, -1.0)
    
    # Aplica rotação (yaw = horizontal, pitch = vertical)
    cos_yaw = ti.cos(yaw)
    sin_yaw = ti.sin(yaw)
    cos_pitch = ti.cos(pitch)
    sin_pitch = ti.sin(pitch)
    
    # Rotação em Y (yaw)
    temp_x = local_dir.x * cos_yaw + local_dir.z * sin_yaw
    temp_z = -local_dir.x * sin_yaw + local_dir.z * cos_yaw
    
    # Rotação em X (pitch)
    dir_x = temp_x
    dir_y = local_dir.y * cos_pitch - temp_z * sin_pitch
    dir_z = local_dir.y * sin_pitch + temp_z * cos_pitch
    
    direction = ti.math.normalize(ti.math.vec3(dir_x, dir_y, dir_z))
    
    return Ray(origin=cam_pos, direction=direction)

@ti.kernel
def render(t: ti.f32):
    """
    Renderiza a cena em TEMPO REAL
    t = tempo (para animação)
    """
    
    # Cena animada: 5 esferas
  
    
    # Esfera 1: Central, muito reflexiva - ÓRBITA
    orbit_radius = 0.8
    angle1 = t * 0.5
    spheres[0].center = ti.math.vec3(
        orbit_radius * ti.cos(angle1),
        0.2 * ti.sin(t * 2.0),  # Bobe verticalmente
        -3.0 + orbit_radius * ti.sin(angle1)
    )
    spheres[0].radius = 0.5
    spheres[0].color = ti.math.vec3(0.9, 0.9, 0.95)
    spheres[0].reflectivity = 0.9
    spheres[0].F0 = 0.8
    
    # Esfera 2: Esquerda - ROTAÇÃO CONTRÁRIA
    angle2 = -t * 0.7
    spheres[1].center = ti.math.vec3(
        -1.2 + 0.3 * ti.cos(angle2),
        -0.3 + 0.15 * ti.sin(t * 1.5),
        -2.5 + 0.3 * ti.sin(angle2)
    )
    spheres[1].radius = 0.45
    spheres[1].color = ti.math.vec3(1.0, 0.3, 0.3)
    spheres[1].reflectivity = 0.6
    spheres[1].F0 = 0.04
    
    # Esfera 3: Direita - MOVIMENTO VERTICAL
    spheres[2].center = ti.math.vec3(
        1.2,
        -0.2 + 0.4 * ti.sin(t * 1.2),
        -2.8
    )
    spheres[2].radius = 0.5
    spheres[2].color = ti.math.vec3(0.3, 1.0, 0.3)
    spheres[2].reflectivity = 0.3
    spheres[2].F0 = 0.04
    
    # Esfera 4: Nova - ÓRBITA RÁPIDA
    angle4 = t * 1.5
    spheres[3].center = ti.math.vec3(
        0.6 * ti.cos(angle4),
        0.5,
        -2.0 + 0.6 * ti.sin(angle4)
    )
    spheres[3].radius = 0.3
    spheres[3].color = ti.math.vec3(1.0, 0.8, 0.2)  # Dourado
    spheres[3].reflectivity = 0.7
    spheres[3].F0 = 0.5
    
    # Esfera 5: Chão (estático)
    spheres[4].center = ti.math.vec3(0.0, -100.5, -3.0)
    spheres[4].radius = 100.0
    spheres[4].color = ti.math.vec3(0.5, 0.5, 0.8)
    spheres[4].reflectivity = 0.4
    spheres[4].F0 = 0.04
    
    # Luz animada (circula)
    light_angle = t * 0.3
    light_pos = ti.math.vec3(
        3.0 * ti.cos(light_angle),
        3.0,
        3.0 * ti.sin(light_angle)
    )
    
    # Obtém posição e ângulos da câmera
    cam_pos = camera_pos[None]
    yaw = camera_angle[0]
    pitch = camera_angle[1]
    
    # Para cada pixel (PARALELO na GPU!)
    for i, j in pixels:
        ray = get_camera_ray(i, j, cam_pos, yaw, pitch)
        color = trace_ray(ray, spheres, num_spheres, light_pos, depth=2)  # 2 bounces para performance
        
        # Correção gamma
        color = ti.math.pow(color, ti.math.vec3(1.0 / 2.2))
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    print("=" * 60)
    print("🚀 RAY TRACING EM TEMPO REAL - GPU M4")
    print("=" * 60)
    print(f"📐 Resolução: {WIDTH}x{HEIGHT}")
    print(f"⚡ Backend: Metal (GPU Apple Silicon)")
    print()
    print("🎮 CONTROLES:")
    print("   W/S - Move câmera frente/trás")
    print("   A/D - Move câmera esquerda/direita")
    print("   Q/E - Move câmera cima/baixo")
    print("   Mouse - Arraste para rotacionar câmera")
    print("   ESC - Sair")
    print("=" * 60)
    print()
    
    # Inicializa câmera
    camera_pos[None] = ti.math.vec3(0.0, 0.5, 1.0)
    camera_angle[0] = 0.0  # yaw
    camera_angle[1] = 0.0  # pitch
    time_var[None] = 0.0
    
    # Cria janela
    window = ti.ui.Window("Ray Tracing GPU - Tempo Real", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    
    # Variáveis de controle
    t = 0.0
    dt = 0.016  # ~60 FPS target
    move_speed = 2.0
    mouse_sensitivity = 0.002
    
    last_mouse_pos = None
    mouse_pressed = False
    
    frame_count = 0
    fps_timer = 0.0
    current_fps = 0.0
    
    print("✅ Renderização iniciada! Feche a janela ou pressione ESC para sair.")
    
    # Loop principal (TEMPO REAL!)
    while window.running:
        # Controles de câmera
        cam_pos = camera_pos[None]
        yaw = camera_angle[0]
        pitch = camera_angle[1]
        
        # Movimento com teclado
        if window.is_pressed('w'):
            cam_pos.z -= move_speed * dt
        if window.is_pressed('s'):
            cam_pos.z += move_speed * dt
        if window.is_pressed('a'):
            cam_pos.x -= move_speed * dt
        if window.is_pressed('d'):
            cam_pos.x += move_speed * dt
        if window.is_pressed('q'):
            cam_pos.y -= move_speed * dt
        if window.is_pressed('e'):
            cam_pos.y += move_speed * dt
        
        # Rotação com mouse
        mouse_pos = window.get_cursor_pos()
        
        if window.is_pressed(ti.ui.LMB):  # Botão esquerdo pressionado
            if last_mouse_pos is not None:
                dx = mouse_pos[0] - last_mouse_pos[0]
                dy = mouse_pos[1] - last_mouse_pos[1]
                
                yaw += dx * mouse_sensitivity * WIDTH
                pitch -= dy * mouse_sensitivity * HEIGHT
                
                # Limita pitch para evitar flip
                pitch = max(-1.5, min(1.5, pitch))
        
        last_mouse_pos = mouse_pos
        
        # Atualiza câmera
        camera_pos[None] = cam_pos
        camera_angle[0] = yaw
        camera_angle[1] = pitch
        
        # Renderiza frame
        render(t)
        
        # Mostra na tela
        canvas.set_image(pixels)
        
        # FPS counter
        frame_count += 1
        fps_timer += dt
        if fps_timer >= 1.0:
            current_fps = frame_count / fps_timer
            print(f"🎯 FPS: {current_fps:.1f} | Tempo: {t:.1f}s | Câmera: ({cam_pos.x:.1f}, {cam_pos.y:.1f}, {cam_pos.z:.1f})", end='\r')
            frame_count = 0
            fps_timer = 0.0
        
        window.show()
        
        # Incrementa tempo
        t += dt
        
        # ESC para sair
        if window.is_pressed(ti.ui.ESCAPE):
            break
    
    print("\n\n✅ Programa finalizado!")

if __name__ == "__main__":
    main()