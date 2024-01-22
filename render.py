import sys
from math import pi
from pathlib import Path
from typing import Union

import bpy


def main():
    input_hdr_file = Path(sys.argv[-1])
    if not input_hdr_file.exists():
        raise FileNotFoundError(input_hdr_file)
    output_filepath = input_hdr_file.parent / f"rendered_{input_hdr_file.stem}.png"
    clean_objects()
    set_camera_object()
    setup_world(input_hdr_file)
    rendering_setting(output_filepath)
    make_global_shadow_catcher_plane()
    make_spheres()
    bpy.ops.render.render(animation=False, write_still=True)

def clean_objects():
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

def clean_nodes(nodes: bpy.types.Nodes):
    for node in nodes:
        nodes.remove(node)

def setup_world(hdri_path: Union[str, Path], hdr_rotation: float = 0):
    bpy.data.worlds["World"].use_nodes = True
    hdri_env_mapping_material_nodes = bpy.data.worlds["World"].node_tree.nodes
    hdri_env_mapping_material_links = bpy.data.worlds["World"].node_tree.links

    clean_nodes(hdri_env_mapping_material_nodes)
    mapping_node = hdri_env_mapping_material_nodes.new(type="ShaderNodeMapping")
    mapping_node.inputs["Rotation"].default_value[2] = hdr_rotation
    tex_coord_node = hdri_env_mapping_material_nodes.new(type="ShaderNodeTexCoord")
    env_texture_node = hdri_env_mapping_material_nodes.new(type="ShaderNodeTexEnvironment")
    env_texture_node.image = bpy.data.images.load(str(hdri_path))
    background_node = hdri_env_mapping_material_nodes.new(type="ShaderNodeBackground")
    output_world_node = hdri_env_mapping_material_nodes.new(type="ShaderNodeOutputWorld")

    hdri_env_mapping_material_links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    hdri_env_mapping_material_links.new(mapping_node.outputs["Vector"], env_texture_node.inputs["Vector"])
    hdri_env_mapping_material_links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
    hdri_env_mapping_material_links.new(background_node.outputs["Background"], output_world_node.inputs["Surface"])

def rendering_setting(
    save_path=Union[str, Path], file_format="PNG", use_gpu=False
):
    scene = bpy.data.scenes["Scene"]
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(save_path)
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.view_settings.view_transform = "Standard"  # Standard is sRGB (default is Filmic) reference: https://blender.stackexchange.com/questions/183699/the-color-of-my-video-is-wrong-when-i-input-it-into-composition

    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_adaptive_threshold = 0.01
    scene.cycles.samples = 10
    NUM_BOUCES = 3
    scene.cycles.max_bounces = NUM_BOUCES
    scene.cycles.diffuse_bounces = NUM_BOUCES
    scene.cycles.glossy_bounces = NUM_BOUCES
    scene.cycles.transmission_bounces = NUM_BOUCES

    scene.cycles.device = "CPU"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "NONE"

def set_camera_object() -> bpy.types.Object:
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera_object = bpy.context.object
    else:
        camera_object = bpy.data.objects["Camera"]
    camera_object.location = (0, -17.3, 10)
    camera_object.rotation_euler = (pi * 63 / 180, 0, 0)
    bpy.context.scene.camera = camera_object
    return camera_object

def make_spheres():
    mat, node = make_sphere_material("1", "ShaderNodeBsdfPrincipled")
    node.inputs[6].default_value = 1.0
    node.inputs[7].default_value = 1.0
    node.inputs[9].default_value = 0.0
    make_sphere([-4.8, 0, 1], mat)

    mat, node = make_sphere_material("2", "ShaderNodeBsdfPrincipled")
    node.inputs[9].default_value = 0.8
    make_sphere([-1.6, 0, 1], mat)

    mat, node = make_sphere_material("3", "ShaderNodeBsdfPrincipled")
    node.inputs[0].default_value = [0,1,1,1]
    node.inputs[7].default_value = 1.0
    node.inputs[9].default_value = 0.3
    make_sphere([1.6, 0, 1], mat)

    mat, node = make_sphere_material("4", "ShaderNodeBsdfGlass")
    node.inputs[2].default_value = 1.6
    make_sphere([4.8, 0, 1], mat)

def make_sphere(location: list, material, scale: float = 1.25):
    bpy.ops.mesh.primitive_uv_sphere_add()
    sphere = bpy.context.selected_objects[0]
    bpy.ops.object.shade_smooth()
    sphere.scale[0] = scale
    sphere.scale[1] = scale
    sphere.scale[2] = scale
    sphere.location[0] = location[0]
    sphere.location[1] = location[1]
    sphere.location[2] = scale
    sphere.data.materials.append(material)

def make_global_shadow_catcher_plane() -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add()
    shadow_catcher_floor_plane = bpy.context.selected_objects[0]

    shadow_catcher_floor_plane.scale[0] = 100
    shadow_catcher_floor_plane.scale[1] = 100
    shadow_catcher_floor_plane.location[0] = 0
    shadow_catcher_floor_plane.location[1] = 0
    shadow_catcher_floor_plane.location[2] = 0

    shadow_catcher_material = make_shadow_catcher_material()
    shadow_catcher_floor_plane.data.materials.append(shadow_catcher_material)
    return shadow_catcher_floor_plane

def make_shadow_catcher_material(mat_name: str = "shadow_catcher_material") -> bpy.types.Material:
    shadow_catcher_material = bpy.data.materials.new(mat_name)
    shadow_catcher_material.use_nodes = True
    shadow_catcher_material_nodes = shadow_catcher_material.node_tree.nodes
    shadow_catcher_material_links = shadow_catcher_material.node_tree.links
    clean_nodes(shadow_catcher_material_nodes)
    output_material_node_room = shadow_catcher_material_nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_diffuse_node_room = shadow_catcher_material_nodes.new(type="ShaderNodeBsdfDiffuse")
    bsdf_diffuse_node_room.inputs[0].default_value = [0.7, 0.7, 0.5, 1.0]
    shadow_catcher_material_links.new(bsdf_diffuse_node_room.outputs["BSDF"], output_material_node_room.inputs["Surface"])
    return shadow_catcher_material

def make_sphere_material(name: str, shader_type: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shadow_catcher_material_nodes = material.node_tree.nodes
    shadow_catcher_material_links = material.node_tree.links

    clean_nodes(shadow_catcher_material_nodes)
    output_material_node = shadow_catcher_material_nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_diffuse_node = shadow_catcher_material_nodes.new(type=shader_type)
    shadow_catcher_material_links.new(bsdf_diffuse_node.outputs["BSDF"], output_material_node.inputs["Surface"])
    return material, bsdf_diffuse_node

if __name__ == "__main__":
    main()
