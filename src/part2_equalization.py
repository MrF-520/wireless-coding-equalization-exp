"""
Part 2：信道均衡实验

学生需要完成 ZF 均衡器估计、FIR 滤波应用和 LMS 自适应均衡。
"""

import numpy as np
from typing import Tuple
from utils import (
    bpsk_demodulate,
    bpsk_modulate,
    calculate_ber,
    generate_bits,
    multipath_channel,
    plot_equalization_results,
    plot_mse_curve,
)


def _build_toeplitz_matrix(channel: np.ndarray, num_taps: int) -> np.ndarray:
    """构造 FIR 卷积矩阵，使 A @ taps 表示 channel 与 taps 的线性卷积。"""
    conv_len = len(channel) + num_taps - 1
    A = np.zeros((conv_len, num_taps), dtype=float)
    for row in range(conv_len):
        for col in range(num_taps):
            idx = row - col
            if 0 <= idx < len(channel):
                A[row, col] = channel[idx]
    return A


def estimate_zf_equalizer(channel: np.ndarray, num_taps: int) -> np.ndarray:
    """
    估计迫零（Zero-Forcing, ZF）FIR 均衡器。

    参数:
        channel: 一维信道冲激响应，例如 np.array([0.9, 0.3, -0.2])。
        num_taps: 均衡器抽头数，建议为奇数。

    返回:
        taps: 一维 FIR 均衡器系数。

    提示:
        1. 构造信道与均衡器卷积的线性方程 A @ taps ≈ d。
        2. d 为中心位置为 1 的冲激响应。
        3. 使用 np.linalg.lstsq 求最小二乘解。
    """
    channel = np.asarray(channel, dtype=float)
    if channel.ndim != 1 or len(channel) == 0:
        raise ValueError('channel 必须是一维非空数组')
    if num_taps < 1:
        raise ValueError('num_taps 必须为正整数')

    # 构造卷积矩阵 A 并生成目标冲激响应 d
    A = _build_toeplitz_matrix(channel, num_taps)
    conv_len = A.shape[0]
    d = np.zeros(conv_len, dtype=float)
    center_pos = (num_taps - 1) // 2
    d[center_pos] = 1.0

    # 使用最小二乘法求解
    taps, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
    return taps


def apply_fir_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """
    对信号应用 FIR 滤波器，并返回与输入等长的输出。

    参数:
        signal: 输入序列。
        taps: FIR 滤波器系数。

    返回:
        filtered: 与 signal 等长的滤波输出。
    """
    signal = np.asarray(signal, dtype=float)
    taps = np.asarray(taps, dtype=float)
    if signal.ndim != 1 or taps.ndim != 1:
        raise ValueError('signal 和 taps 必须是一维数组')

    # 使用 np.convolve 进行卷积
    filtered = np.convolve(signal, taps, mode='full')
    
    # 截取前 len(signal) 个样本
    filtered = filtered[:len(signal)]
    
    return filtered


def lms_equalizer(rx_train: np.ndarray, tx_train: np.ndarray, num_taps: int, step_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用训练序列实现 LMS 自适应均衡。

    参数:
        rx_train: 接收训练序列。
        tx_train: 期望发送训练符号。
        num_taps: 均衡器抽头数。
        step_size: LMS 步长 μ。

    返回:
        taps: 训练后的均衡器系数。
        errors: 每次迭代的误差 e[n]。

    提示:
        1. 抽头向量可初始化为中心抽头为 1。
        2. y[n] = w^T x[n]
        3. e[n] = d[n] - y[n]
        4. w = w + μ e[n] x[n]
    """
    rx_train = np.asarray(rx_train, dtype=float)
    tx_train = np.asarray(tx_train, dtype=float)
    if len(rx_train) != len(tx_train):
        raise ValueError('rx_train 和 tx_train 长度必须一致')
    if num_taps < 1:
        raise ValueError('num_taps 必须为正整数')

    # 初始化 taps，中心抽头为 1，其余为 0
    taps = np.zeros(num_taps)
    center_pos = (num_taps - 1) // 2
    taps[center_pos] = 1.0
    
    errors = []
    
    # 从第 num_taps - 1 个样本开始迭代
    for n in range(num_taps - 1, len(rx_train)):
        # 构造当前输入向量 x，长度为 num_taps
        # x 为最新的 num_taps 个接收样本（从后向前排列）
        x = rx_train[n - num_taps + 1:n + 1][::-1]
        
        # 计算输出 y = taps @ x
        y = taps @ x
        
        # 计算误差 e = d - y，d 来自 tx_train
        e = tx_train[n] - y
        
        # 根据 LMS 公式更新：taps = taps + step_size * e * x
        taps = taps + step_size * e * x
        
        # 保存每次迭代的误差
        errors.append(e)
    
    errors = np.array(errors)
    return taps, errors


def run_equalization_demo():
    """运行 Part 2 演示并生成均衡效果图。"""
    print('=' * 60)
    print('Part 2：信道均衡实验')
    print('=' * 60)

    try:
        bits = generate_bits(2000, seed=2027)
        symbols = bpsk_modulate(bits)
        channel = np.array([0.9, 0.35, -0.25])
        rx = multipath_channel(symbols, channel, noise_std=0.12, seed=7)

        zf_taps = estimate_zf_equalizer(channel, num_taps=7)
        zf_output = apply_fir_filter(rx, zf_taps)

        lms_taps, errors = lms_equalizer(rx[:800], symbols[:800], num_taps=7, step_size=0.01)
        lms_output = apply_fir_filter(rx, lms_taps)

        raw_bits = bpsk_demodulate(rx[: len(bits)])
        eq_bits = bpsk_demodulate(lms_output[: len(bits)])
        print(f'均衡前 BER: {calculate_ber(bits, raw_bits):.4f}')
        print(f'LMS 均衡后 BER: {calculate_ber(bits, eq_bits):.4f}')

        plot_equalization_results(symbols, rx, lms_output, 'equalization_eye_comparison.png')
        plot_mse_curve(errors, 'equalization_mse_curve.png')
        print('✅ 已生成均衡结果图')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 2 运行失败：{error}')


if __name__ == '__main__':
    run_equalization_demo()
