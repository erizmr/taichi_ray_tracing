# 太极图形课S1-Ray Tracing示例程序

## 背景简介
该repo实现了一些Rendering的方法, Path tracing的具体实现参考了[Ray Tracing in One Weekend](https://raytracing.github.io/)


## 效果展示
|Color only |Lambertian reflection | Blinn-Phong model |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="./img/color_only.png" width="200"> | <img src="./img/lambertian.png" width="200"> |<img src="./img/b_p.png" width="200"> |
|Blinn-Phong model with shadow| Whitted style ray tracing|Path tracing|
|<img src="./img/b_p_with_shadow.png" width="200">|<img src="./img/whitted_style.png" width="200">|<img src="./img/path_tracing_sample_on_sphere_surface.png" width="200">|


## 运行环境

```
[Taichi] version 0.8.3, llvm 10.0.0, commit 2680dabd, linux, python 3.8.10
```

## 运行方式
确保ray_tracing_models.py可以访问的情况下，可以直接运行：`python3 [*].py`
