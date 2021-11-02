import taichi as ti
import numpy as np
import argparse
from ray_tracing_models import Ray, Camera, Hittable_list, Sphere, PI, random_in_unit_sphere
ti.init(arch=ti.cpu, cpu_max_num_threads=5)

PI = 3.14159265

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

num_of_rays = 10
scattered_origin_buffer = ti.Vector.field(3, dtype=float, shape=num_of_rays)
scattered_direction_buffer = ti.Vector.field(3, dtype=float, shape=num_of_rays)
scattered_buffer_mask = ti.field(dtype=int, shape=num_of_rays)


# Rendering parameters
samples_per_pixel = 4
max_depth = 10

@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray)
        color /= samples_per_pixel
        canvas[i, j] += color

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

# @ti.func
# def add_shadow(hit_point, hit_point_to_source):
#     ret = 0.0
#     # Add shadow
#     is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric = scene.hit_shadow(
#         Ray(hit_point, hit_point_to_source))
#     if not is_hit_source:
#         if is_hitted_non_dielectric:
#             # Add hard shadow
#             ret = 0.0
#         elif hitted_dielectric_num > 0:
#             # Add soft shadow if the obstacles are dielectric
#             ret = ti.pow(0.5, hitted_dielectric_num)
#     return ret
@ti.func
def clean_buffer():
    for i in range(num_of_rays):
        scattered_buffer_mask[i] = 0
        scattered_origin_buffer[i] = ti.Vector([0.0, 0.0, 0.0])
        scattered_direction_buffer[i] = ti.Vector([0.0, 0.0, 0.0])

# Whitted-style ray tracing
@ti.func
def ray_color(ray):
    default_color = ti.Vector([0.0, 0.0, 0.0])
    specular_color_buffer = ti.Vector([0.0, 0.0, 0.0])
    diffuse_color_buffer = ti.Vector([0.0, 0.0, 0.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction

    clean_buffer()
    scattered_origin_buffer[0] = scattered_origin
    scattered_direction_buffer[0] = scattered_direction
    scattered_buffer_mask[0] = 1
    buffer_pointer = 0

    is_hit = False
    front_face = False
    material = 1
    hit_point = ti.Vector([0.0, 0.0, 0.0])
    hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
    color = ti.Vector([0.0, 0.0, 0.0])

    for i in range(3):
        if scattered_buffer_mask[i] == 1:
            scattered_origin = scattered_origin_buffer[i]
            scattered_direction = scattered_direction_buffer[i]
            for n in range(max_depth):
                is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
                if is_hit:
                    if material == 0:
                        default_color = color
                        break
                    else:
                        # Compute the local color use Blinn-Phong model
                        hit_point_to_source = to_light_source(hit_point, ti.Vector([0, 5.4 - 3.0, -1]))
                        # Diffuse light
                        local_color = color * max(
                            hit_point_to_source.dot(hit_point_normal) / (
                                    hit_point_to_source.norm() * hit_point_normal.norm()),
                            0.0)
                        # Diffuse
                        if material == 1:
                            diffuse_color_buffer = local_color
                            # Add shadow
                            is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric = scene.hit_shadow(
                                Ray(hit_point, hit_point_to_source))
                            if not is_hit_source:
                                if is_hitted_non_dielectric:
                                    # Add hard shadow
                                    diffuse_color_buffer *= 0
                                elif hitted_dielectric_num > 0:
                                    # Add soft shadow if the obstacles are dielectric
                                    diffuse_color_buffer *= ti.pow(0.5, hitted_dielectric_num)
                            break
                        else:
                            intensity = 0.0
                            # Specular light
                            H = (-(scattered_direction.normalized()) + hit_point_to_source.normalized()).normalized()
                            N_dot_H = max(H.dot(hit_point_normal.normalized()), 0.0)
                            intensity = ti.pow(N_dot_H, 10)
                            specular_color = intensity * color
                            specular_color_buffer += 0.1 * specular_color + 0.9 * color

                            # Add shadow
                            is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric = scene.hit_shadow(
                                Ray(hit_point, hit_point_to_source))
                            if not is_hit_source:
                                if is_hitted_non_dielectric:
                                    # Add hard shadow
                                    specular_color_buffer *= 0
                                elif hitted_dielectric_num > 0:
                                    # Add soft shadow if the obstacles are dielectric
                                    specular_color_buffer *= ti.pow(0.5, hitted_dielectric_num)
                            # Metal
                            if material == 2 or material == 4:
                                scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal) # + 0.4 * random_in_unit_sphere()
                                scattered_origin = hit_point
                                if scattered_direction.dot(hit_point_normal) < 0:
                                    break
                            # Dieletric
                            elif material == 3:
                                refraction_ratio = 1.5
                                if front_face:
                                    refraction_ratio = 1 / refraction_ratio
                                cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                                sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                                # total internal reflection
                                if refraction_ratio * sin_theta > 1.0: # or reflectance(cos_theta, refraction_ratio) > ti.random():
                                    scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                                else:
                                    scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                                scattered_origin = hit_point
            # scattered_direction_buffer[i] = scattered_direction
            # scattered_origin_buffer[i] = scattered_origin
    default_color += 0.5 * specular_color_buffer + diffuse_color_buffer
    return default_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive Ray Tracing')
    parser.add_argument(
        '--max_depth', type=int, default=10, help='max depth (default: 10)')
    parser.add_argument(
        '--samples_per_pixel', type=int, default=4, help='samples_per_pixel  (default: 4)')
    args = parser.parse_args()

    max_depth = args.max_depth
    samples_per_pixel = args.samples_per_pixel
    scene = Hittable_list()

    # Light source
    scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # right wall
    scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    # Diffuse ball
    scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
    # Metal ball
    scene.add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
    # Glass ball
    scene.add(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # Metal ball-2
    scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])))

    camera = Camera()
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    canvas.fill(0)
    cnt = 0
    while gui.running:
        render()
        cnt += 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))
        gui.show()