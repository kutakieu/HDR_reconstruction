from typing import List, Literal, Tuple, Union

import cv2
import numpy as np

InterpolationMode = Literal["bilinear", "nearest"]
CubeFormat = Literal["horizon", "list", "dict", "dice"]
interpolation_mode2order = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
}


def e2p(e_img: np.ndarray, 
        fov_deg: Union[Tuple[float, float], float], 
        u_deg: float, 
        v_deg: float, 
        out_hw: Tuple[int, int], 
        in_rot_deg: float=0, 
        mode: InterpolationMode='bilinear')->np.ndarray:
    '''
    e_img:   ndarray in shape of [H, W, *]
    fov_deg: scalar or (scalar[h_fov], scalar[v_fov]) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    if mode not in interpolation_mode2order:
        raise NotImplementedError('unknown interpolation mode')

    if len(e_img.shape) == 2:
        e_img = e_img[..., np.newaxis]
    h, w = e_img.shape[:2]

    coor_xy = generate_e2p_map((h, w), fov_deg, u_deg, v_deg, out_hw, in_rot_deg)

    pers_img = _sample_equirec(e_img, coor_xy, interpolation=interpolation_mode2order[mode])

    return pers_img

def generate_e2p_map(
        in_hw: Tuple[int, int],
        fov_deg: Union[Tuple[float, float], float], 
        u_deg: float, 
        v_deg: float, 
        out_hw: Tuple[int, int], 
        in_rot_deg: float=0
        )->np.ndarray:
    h, w = in_hw
    if isinstance(fov_deg, (int, float)):
        fov_deg = (fov_deg, fov_deg)
    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    in_rot = in_rot_deg * np.pi / 180

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = _xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = _xyz2uv(xyz)
    return _uv2coor(uv, h, w)

def e2p_with_map(e_img: np.ndarray, e2p_map: np.ndarray, mode: InterpolationMode='bilinear'):
    if mode not in interpolation_mode2order:
        raise NotImplementedError('unknown interpolation mode')

    if len(e_img.shape) == 2:
        e_img = e_img[..., np.newaxis]

    return _sample_equirec(e_img, e2p_map, interpolation=interpolation_mode2order[mode])

def _sample_equirec(e_img: np.ndarray, coor_xy: np.ndarray, interpolation: int):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return cv2.remap(e_img, 
                     coor_x.astype(np.float32), 
                     coor_y.astype(np.float32), 
                     interpolation=interpolation, 
                     borderMode=cv2.BORDER_WRAP)

def _uv2coor(uv: np.ndarray, h: int, w: int):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)

def _xyz2uv(xyz: np.ndarray):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)

def _xyzpers(h_fov: float, 
            v_fov: float, 
            u: float, v: float, 
            out_hw: Tuple[int, int], 
            in_rot: float):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = _rotation_matrix(v, [1, 0, 0])
    Ry = _rotation_matrix(u, [0, 1, 0])
    Ri = _rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))

    return out.dot(Rx).dot(Ry).dot(Ri)

def _rotation_matrix(rad: float, ax: List[int]):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
    return R
