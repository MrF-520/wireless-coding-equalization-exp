"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from typing import Optional, Tuple
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)

CONVOLUTIONAL_GENERATORS = (
    (1, 1, 1),
    (1, 0, 1),
)


def _validate_binary_array(bits: np.ndarray, name: str) -> np.ndarray:
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError(f'{name} 必须是一维数组')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError(f'{name} 只能包含 0 或 1')
    return bits


def _syndrome_to_error_position(syndrome: np.ndarray) -> Optional[int]:
    for bit_pos in range(7):
        if np.array_equal(syndrome, HAMMING_H[:, bit_pos]):
            return bit_pos
    return None


def _convolutional_output_bits(input_bit: int, state: int) -> Tuple[int, int, int]:
    register = (input_bit, (state >> 1) & 1, state & 1)
    g1, g2 = CONVOLUTIONAL_GENERATORS
    out1 = (register[0] * g1[0] + register[1] * g1[1] + register[2] * g1[2]) % 2
    out2 = (register[0] * g2[0] + register[1] * g2[1] + register[2] * g2[2]) % 2
    next_state = ((state << 1) | input_bit) & 0x3
    return out1, out2, next_state


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = _validate_binary_array(bits, 'bits')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')

    # 将 bits reshape 为 (-1, 4)
    blocks = bits.reshape(-1, 4)
    
    # 与 HAMMING_G 相乘并对 2 取模
    encoded_blocks = (blocks @ HAMMING_G) % 2
    
    # flatten 成一维数组返回
    encoded = encoded_blocks.flatten()
    
    return encoded


def hamming74_syndrome(codewords: np.ndarray) -> np.ndarray:
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    # 计算 s = codewords @ H^T mod 2
    syndromes = (codewords @ HAMMING_H.T) % 2
    
    return syndromes


def hamming74_decode(received: np.ndarray) -> np.ndarray:
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    # 将 received reshape 为 (-1, 7)，复制一份避免直接修改输入
    codewords = received.reshape(-1, 7).copy()
    
    # 计算伴随式
    syndromes = hamming74_syndrome(codewords)
    
    # 对每个码字进行纠错
    for i, syndrome in enumerate(syndromes):
        if np.any(syndrome):
            bit_pos = _syndrome_to_error_position(syndrome)
            if bit_pos is not None:
                codewords[i, bit_pos] = 1 - codewords[i, bit_pos]
    
    # 取每个码字前 4 位作为信息比特，flatten 返回
    decoded_bits = codewords[:, :4].flatten()
    
    return decoded_bits


def convolutional_encode(bits: np.ndarray) -> np.ndarray:
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = _validate_binary_array(bits, 'bits')

    bits_with_tail = np.concatenate([bits, np.zeros(2, dtype=int)])
    encoded = []
    state = 0  # 初始状态为 0

    for bit in bits_with_tail:
        out1, out2, state = _convolutional_output_bits(int(bit), state)
        encoded.append(out1)
        encoded.append(out2)

    return np.array(encoded, dtype=int)


def viterbi_decode_hard(received_bits: np.ndarray) -> np.ndarray:
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。
    """
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    num_states = 4  # 2^2 个状态
    num_symbols = len(received_bits) // 2

    path_metrics = np.full(num_states, np.inf)
    path_metrics[0] = 0.0

    predecessors = np.zeros((num_symbols, num_states), dtype=int)
    input_bits_seq = np.zeros((num_symbols, num_states), dtype=int)

    for symbol_idx in range(num_symbols):
        received = received_bits[2 * symbol_idx:2 * symbol_idx + 2]
        new_path_metrics = np.full(num_states, np.inf)

        for curr_state in range(num_states):
            if path_metrics[curr_state] == np.inf:
                continue

            for input_bit in [0, 1]:
                out1, out2, next_state = _convolutional_output_bits(input_bit, curr_state)
                hamming_dist = abs(out1 - received[0]) + abs(out2 - received[1])
                new_metric = path_metrics[curr_state] + hamming_dist

                if new_metric < new_path_metrics[next_state]:
                    new_path_metrics[next_state] = new_metric
                    predecessors[symbol_idx, next_state] = curr_state
                    input_bits_seq[symbol_idx, next_state] = input_bit

        path_metrics = new_path_metrics

    final_state = np.argmin(path_metrics)
    decoded = []
    current_state = final_state

    for symbol_idx in range(num_symbols - 1, -1, -1):
        input_bit = input_bits_seq[symbol_idx, current_state]
        decoded.append(input_bit)
        current_state = predecessors[symbol_idx, current_state]

    decoded.reverse()
    return np.array(decoded[:-2], dtype=int)


def run_coding_demo() -> None:
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
        
        # 卷积码演示
        print('\n' + '=' * 60)
        print('卷积码 (2,1,3) 与 Viterbi 硬判决译码演示')
        print('=' * 60)
        
        conv_ber = []
        bits_conv = generate_bits(4000, seed=2027)
        conv_encoded = convolutional_encode(bits_conv)
        
        for index, probability in enumerate(error_probabilities):
            conv_rx = binary_symmetric_channel(conv_encoded, probability, seed=300 + index)
            conv_decoded = viterbi_decode_hard(conv_rx)
            
            # 截断到原始长度进行 BER 计算
            conv_ber.append(calculate_ber(bits_conv[:len(conv_decoded)], conv_decoded))
        
        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber, '卷积码(2,1,3)': conv_ber},
            '信道编码方案 BER 对比',
            'coding_comparison_ber_curve.png',
        )
        print('✅ 已生成 results/coding_comparison_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')



if __name__ == '__main__':
    run_coding_demo()
