import taichi as ti
import numpy as np
import argparse
from ray_tracing_models import Ray, Camera, Hittable_list, Sphere, PI, random_in_unit_sphere, refract, reflect, reflectance, to_light_source
ti.init(arch=ti.gpu)

PI = 3.14159265
# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
light_source = ti.Vector([0, 5.4 - 3.0, -1])

# Rendering parameters
samples_per_pixel = 4
stack_depth = 10

@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray, i, j)
        color /= samples_per_pixel
        canvas[i, j] += color

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point


@ti.func
def blinn_phong(ray_direction, hit_point, hit_point_normal, color, material):
    # Compute the local color use Blinn-Phong model
    hit_point_to_source = to_light_source(hit_point, light_source)
    # Diffuse light
    diffuse_color = color * ti.max(
        hit_point_to_source.dot(hit_point_normal) / (
                hit_point_to_source.norm() * hit_point_normal.norm()),
        0.0)
    specular_color = ti.Vector([0.0, 0.0, 0.0])
    diffuse_weight = 1.0
    specular_weight = 1.0
    if material != 1:
        # Specular light
        H = (-(ray_direction.normalized()) + hit_point_to_source.normalized()).normalized()
        N_dot_H = ti.max(H.dot(hit_point_normal.normalized()), 0.0)
        intensity = ti.pow(N_dot_H, 10)
        specular_color = intensity * color

    # Dieletric
    if material == 3:
        diffuse_weight = 0.1
    # Fuzz metal ball
    if material == 4:
        diffuse_weight = 0.5
        specular_weight = 0.5

    # Add shadow
    is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric = scene.hit_shadow(
        Ray(hit_point, hit_point_to_source))
    shadow_weight = 1.0
    if not is_hit_source:
        if is_hitted_non_dielectric:
            # Add hard shadow
            shadow_weight = 0
        elif hitted_dielectric_num > 0:
            # Add soft shadow if the obstacles are dielectric
            shadow_weight = ti.pow(0.5, hitted_dielectric_num)

    return (diffuse_weight * diffuse_color + specular_weight * specular_color) * shadow_weight


origin_stack = ti.Vector.field(3, dtype=float, shape=(image_width, image_height, stack_depth))
origin_stack_pointer = ti.field(dtype=int, shape=(image_width, image_height))

direction_stack = ti.Vector.field(3, dtype=float, shape=(image_width, image_height, stack_depth))
direction_stack_pointer = ti.field(dtype=int, shape=(image_width, image_height))

reflect_refract_stack = ti.Vector.field(2, dtype=int, shape=(image_width, image_height, stack_depth))
reflect_refract_stack_pointer = ti.field(dtype=int, shape=(image_width, image_height))

color_weight_stack = ti.field(dtype=float, shape=(image_width, image_height, stack_depth))
color_weight_stack_pointer = ti.field(dtype=int, shape=(image_width, image_height))

@ti.func
def stack_clear(i, j):
    origin_stack_pointer[i, j] = -1
    direction_stack_pointer[i, j] = -1
    reflect_refract_stack_pointer[i, j] = -1
    color_weight_stack_pointer[i, j] = -1

@ti.func
def stack_push(i, j, hit_point, new_direction, color_weight):
    origin_stack_pointer[i, j] += 1
    direction_stack_pointer[i, j] += 1
    reflect_refract_stack_pointer[i, j] += 1
    color_weight_stack_pointer[i, j] += 1

    origin_stack[i, j, origin_stack_pointer[i, j]] = hit_point
    direction_stack[i, j, direction_stack_pointer[i, j]] = new_direction
    reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]] = ti.Vector([0, 0])
    color_weight_stack[i, j, color_weight_stack_pointer[i, j]] = color_weight

@ti.func
def stack_pop(i, j):
    origin_stack_pointer[i, j] -= 1
    direction_stack_pointer[i, j] -= 1
    reflect_refract_stack_pointer[i, j] -= 1
    color_weight_stack_pointer[i, j] -= 1

@ti.func
def stack_top(i, j):
    return origin_stack[i, j, origin_stack_pointer[i, j]], \
           direction_stack[i, j, direction_stack_pointer[i, j]], \
           reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]], \
           color_weight_stack[i, j, color_weight_stack_pointer[i, j]]


# Whitted-style ray tracing
@ti.func
def ray_color(ray, i, j):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    color_buffer_temp = ti.Vector([0.0, 0.0, 0.0])
    stack_clear(i, j)
    stack_push(i, j, ray.origin, ray.direction, 1.0)
    while origin_stack_pointer[i, j] >= 0 and origin_stack_pointer[i, j] < stack_depth:
        # Fetch a ray
        curr_origin, curr_direction, curr_relect_refract, color_weight = stack_top(i, j)
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(curr_origin, curr_direction))
        if is_hit:
            # Light source
            if material == 0:
                color_buffer = color * color_weight
                # Pop
                stack_pop(i, j)
            # Diffuse
            elif material == 1:
                local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                color_buffer = local_color * color_weight
                # Pop
                stack_pop(i, j)
            # Metal
            elif material == 2 or material == 4:
                fuzz = 0.0
                if material == 4:
                    fuzz = 0.4
                refected = reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0]
                if not refected:
                    # Reflect
                    reflected_direction = reflect(curr_direction.normalized(), hit_point_normal) + fuzz * random_in_unit_sphere()
                    reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0] = 1
                    if reflected_direction.dot(hit_point_normal) > 0:
                        stack_push(i, j, hit_point, reflected_direction, 1.0)
                else:
                    local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                    color_buffer += local_color
                    stack_pop(i, j)
            # Dieletric
            elif material == 3:
                refraction_ratio = 1.5
                if front_face:
                    refraction_ratio = 1 / refraction_ratio
                cos_theta = min(-curr_direction.normalized().dot(hit_point_normal), 1.0)
                sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                reflect_weight = reflectance(cos_theta, refraction_ratio)
                refract_weight = 1- reflect_weight

                refected = reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0]
                refracted = reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][1]
                if not refected:
                    # Reflect
                    reflected_direction = reflect(curr_direction.normalized(), hit_point_normal) # + fuzz * random_in_unit_sphere()
                    reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][0] = 1
                    stack_push(i, j, hit_point, reflected_direction, reflect_weight)
                else:
                    local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                    color_buffer = color_buffer + 0.1 * local_color
                    stack_pop(i, j)

                if not refracted:
                    # Check total internal reflection
                    if refraction_ratio * sin_theta <= 1.0:
                        # Refract
                        refracted_direction = refract(curr_direction.normalized(), hit_point_normal, refraction_ratio)
                        reflect_refract_stack[i, j, reflect_refract_stack_pointer[i, j]][1] = 1
                        stack_push(i, j, hit_point, refracted_direction, refract_weight)
                else:
                    local_color = blinn_phong(curr_direction, hit_point, hit_point_normal, color, material)
                    color_buffer = color_buffer + 0.1 * local_color
                    stack_pop(i, j)
            else:
                stack_pop(i, j)
        else:
            stack_pop(i, j)

    return color_buffer

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