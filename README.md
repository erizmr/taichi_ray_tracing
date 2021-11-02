# 太极图形课S1-Ray Tracing示例程序

## 背景简介
该repo实现了一些Rendering的方法, Path tracing的具体实现参考了[Ray Tracing in One Weekend](https://raytracing.github.io/)


## 效果展示
### Color only
![Color only](./img/color_only.png)

### Lambertian reflection 
![Lambertian reflection](./img/lambertian.png)

### Blinn-Phong model
![Blinn-Phong model](./img/b_p.png)

### Blinn-Phong model with shadow
![Blinn-Phong model with shadow](./img/b_p_with_shadow.png)

### Whitted style ray tracing
![Whitted style ray tracing](./img/whitted.png)

### Path tracing
![Path tracing](./img/path_tracing.png)


## 运行环境

```
[Taichi] version 0.8.3, llvm 10.0.0, commit 2680dabd, linux, python 3.8.10
```

## 运行方式
确保ray_tracing_models.py可以访问的情况下，可以直接运行：`python3 [*].py`
