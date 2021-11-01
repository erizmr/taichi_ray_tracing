import taichi as ti
import numpy as np
import math
import argparse

ti.init(arch=ti.gpu)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))


# Rendering parameters
samples_per_pixel = 4
max_depth = 10

# Camera parameters
lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
up = ti.Vector.field(3, dtype=ti.f32, shape=())
fov = 60

cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

# objects
sphere_num = 10
sphere_origin_list = ti.Vector.field(3, dtype=ti.f32, shape=(sphere_num))  # Center
sphere_radius_list = ti.field(dtype=ti.f32, shape=(sphere_num))  # Radius
sphere_material_list = ti.field(dtype=ti.i32,
                              shape=(sphere_num))  # Material，0=light source, 1=diffuse，2=Metal，3=dielectric
sphere_material_color_list = ti.Vector.field(3, dtype=ti.f32,
                                       shape=(sphere_num))  # color
sphere_metal_fuzz_list = ti.field(dtype=ti.f32, shape=(sphere_num))  # fuzz


# Light
sphere_origin_list[0] = [0, 5.4, -1]
sphere_radius_list[0] = 3
sphere_material_list[0] = 0
sphere_material_color_list[0] = [10, 10, 10]

# ground
sphere_origin_list[1] = [0, -100.5, -1]
sphere_radius_list[1] = 100
sphere_material_list[1] = 1
sphere_material_color_list[1] = [0.8, 0.8, 0.8]

# ceiling
sphere_origin_list[2] = [0, 102.5, -1]
sphere_radius_list[2] = 100
sphere_material_list[2] = 1
sphere_material_color_list[2] = [0.8, 0.8, 0.8]

# back wall
sphere_origin_list[3] = [0, 1, 101]
sphere_radius_list[3] = 100
sphere_material_list[3] = 1
sphere_material_color_list[3] = [0.8, 0.8, 0.8]

# right wall
sphere_origin_list[4] = [-101.5, 0, -1]
sphere_radius_list[4] = 100
sphere_material_list[4] = 1
sphere_material_color_list[4] = [0.6, 0.0, 0.0]

# left wall
sphere_origin_list[5] = [101.5, 0, -1]
sphere_radius_list[5] = 100
sphere_material_list[5] = 1
sphere_material_color_list[5] = [0.0, 0.6, 0.0]

# Diffuse ball
sphere_origin_list[6] = [0, -0.2, -1.5]
sphere_radius_list[6] = 0.3
sphere_material_list[6] = 1
sphere_material_color_list[6] = [0.8, 0.3, 0.3]

# Metal ball 1
sphere_origin_list[7] = [-0.8, 0.2, -1]
sphere_radius_list[7] = 0.7
sphere_material_list[7] = 2
sphere_material_color_list[7] = [0.6, 0.8, 0.8]
sphere_metal_fuzz_list[7] = 0.0

# Glass ball 1
sphere_origin_list[8] = [0.7, 0, -0.5]
sphere_radius_list[8] = 0.5
sphere_material_list[8] = 3
sphere_material_color_list[8] = [1.0, 1.0, 1.0]
sphere_metal_fuzz_list[8] = 1.5

# Metal ball 2
sphere_origin_list[9] = [0.6, -0.3, -2.0]
sphere_radius_list[9] = 0.2
sphere_material_list[9] = 2
sphere_material_color_list[9] = [0.8, 0.6, 0.2]
sphere_metal_fuzz_list[9] = 0.4

lookfrom[None] = [0.0, 1.0, -5.0]


@ti.kernel
def generate_cam_parameter():
    lookat[None] = [0.0, 1.0, -1.0]
    up[None] = [0.0, 1.0, 0.0]
    theta = fov * (math.pi / 180.0)
    half_height = ti.tan(theta / 2.0)
    half_width = aspect_ratio * half_height
    cam_origin[None] = lookfrom[None]
    w = (lookfrom[None] - lookat[None]).normalized()
    u = (up[None].cross(w)).normalized()
    v = w.cross(u)
    cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
    cam_lower_left_corner[
        None] = cam_origin[None] - half_width * u - half_height * v - w
    cam_horizontal[None] = 2 * half_width * u
    cam_vertical[None] = 2 * half_height * v


@ti.func
def random_in_unit_sphere():
    theta = 2.0 * math.pi * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)
    r = ti.pow(ti.random(), 1.0/3.0)
    return ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi)])


@ti.func
def cam_get_ray(u, v):
    return cam_origin[None], cam_lower_left_corner[
        None] + u * cam_horizontal[None] + v * cam_vertical[None] - cam_origin[None]

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(1.0 - r_out_perp.dot(r_out_perp)) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.func
def set_face_normal(ray_direction, outward_normal):
    front_face = (ray_direction.dot(outward_normal) < 0)
    normal = outward_normal
    if not front_face:
        normal = -outward_normal
    return front_face, normal

@ti.func
def scatter(ray_origin, ray_direction, hit_p, hit_normal, hit_material,
            hit_fuzz, front_face):
    scattered_ray_origin = hit_p
    scattered_ray_direction = ti.Vector([0.0, 0.0, 0.0])
    flag = False
    if hit_material == 1:  # Diffuse
        target = hit_p + hit_normal + random_in_unit_sphere()
        scattered_ray_direction = target - hit_p
        flag = True
    elif hit_material == 2:  # Metal
        reflected = reflect((ray_direction).normalized(), hit_normal)
        scattered_ray_direction = reflected + hit_fuzz * random_in_unit_sphere(
        )
        flag = scattered_ray_direction.dot(hit_normal) > 0
    elif hit_material == 3: # Dielectrics
        refraction_ratio = hit_fuzz # Here the fuzz is refraction ratio for dielectrics
        if front_face:
            refraction_ratio = 1.0 / hit_fuzz
        cos_theta = min(hit_normal.dot(-(ray_direction).normalized()), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
        if (refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random()):
            scattered_ray_direction = reflect((ray_direction).normalized(), hit_normal)
        else:
            scattered_ray_direction = refract((ray_direction).normalized(), hit_normal, refraction_ratio)
        flag = True
    return flag, scattered_ray_origin, scattered_ray_direction


@ti.func
def hit_sphere(sphere_center, sphere_radius, ray_origin, ray_direction, t_min,
               t_max):
    oc = ray_origin - sphere_center
    a = ray_direction.dot(ray_direction)
    b = oc.dot(ray_direction)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    discriminant = b * b - a * c

    hit_flag = False
    hit_t = 0.0
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    front_face = False
    if discriminant > 0.0:
        temp = (-b - ti.sqrt(b * b - a * c)) / a
        if temp < t_max and temp > t_min:
            hit_t = temp
            hit_p = ray_origin + hit_t * ray_direction
            hit_normal = (hit_p - sphere_center) / sphere_radius
            hit_flag = True
        if hit_flag == False:
            temp = (-b + ti.sqrt(b * b - a * c)) / a
            if temp < t_max and temp > t_min:
                hit_t = temp
                hit_p = ray_origin + hit_t * ray_direction
                hit_normal = (hit_p - sphere_center) / sphere_radius
                hit_flag = True
    if hit_flag:
        front_face, hit_normal = set_face_normal(ray_direction, hit_normal)
    return hit_flag, hit_t, hit_p, hit_normal, front_face


@ti.func
def hit_all_spheres(ray_origin, ray_direction, t_min, t_max):
    hit_anything = False
    hit_t = 0.0
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_material = 1
    hit_material_color = ti.Vector([0.0, 0.0, 0.0])
    hit_fuzz = 0.0
    front_face = False
    closest_so_far = t_max
    for i in range(sphere_num):
        hit_flag, temp_hit_t, temp_hit_p, temp_hit_normal, temp_front_face = \
            hit_sphere(sphere_origin_list[i], sphere_radius_list[i], ray_origin, ray_direction, t_min, closest_so_far)
        if hit_flag:
            hit_anything = True
            closest_so_far = temp_hit_t
            hit_t = temp_hit_t
            hit_p = temp_hit_p
            hit_normal = temp_hit_normal
            hit_material = sphere_material_list[i]
            hit_material_color = sphere_material_color_list[i]
            hit_fuzz = sphere_metal_fuzz_list[i]
            front_face = temp_front_face
    return hit_anything, hit_t, hit_p, hit_normal, hit_material, hit_material_color, hit_fuzz, front_face


@ti.func
def color(ray_origin, ray_direction):
    col = ti.Vector([0.0, 0.0, 0.0])
    coefficient = ti.Vector([1.0, 1.0, 1.0])
    for i in range(max_depth):
        hit_flag, hit_t, hit_p, hit_normal, hit_material, hit_material_color, hit_fuzz, front_face = \
            hit_all_spheres(ray_origin, ray_direction, 0.001, 10e9)
        if hit_flag:
            if hit_material == 0:  # light source
                col = coefficient * hit_material_color
                break
            flag, ray_origin, ray_direction = \
                scatter(ray_origin, ray_direction, hit_p, hit_normal, hit_material, hit_fuzz, front_face)
            if flag:
                coefficient *= hit_material_color
            else:
                break
        # else:
        #     unit_direction = (ray_direction).normalized()
        #     t = 0.5 * (unit_direction.y + 1.0)
        #     # col = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector(
        #     #     [0.5, 0.7, 1.0])
        #     col = (1.0 - t) * ti.Vector([0.0, 0.0, 0.0]) + t * ti.Vector(
        #         [0.0, 0.0, 0.0])
        #     col *= coefficient
        #     break
    return col


@ti.kernel
def draw():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        col = ti.Vector.zero(ti.f32, 3)
        for t in ti.static(range(samples_per_pixel)):
            ray_origin, ray_direction = cam_get_ray(u, v)
            col += color(ray_origin, ray_direction)
        canvas[i, j] += col / samples_per_pixel

gui = ti.GUI("Naive Ray Tracing", (image_width, image_height))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive Ray Tracing')
    parser.add_argument(
        '--max_depth', type=int, default=10, help='max depth (default: 10)')
    parser.add_argument(
        '--samples_per_pixel', type=int, default=4, help='samples_per_pixel  (default: 4)')
    args = parser.parse_args()

    max_depth = args.max_depth
    samples_per_pixel = args.samples_per_pixel

    generate_cam_parameter()
    canvas.fill(0)
    cnt = 0
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.LMB:
                x, y = gui.get_cursor_pos()
                lookfrom[None][0] = x * 4 - 2
                lookfrom[None][1] = y * 2 - 1
                cnt = 0
                canvas.fill(0)
                generate_cam_parameter()
            elif gui.event.key == ti.GUI.ESCAPE:
                exit()
        draw()
        cnt += 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))
        gui.show()