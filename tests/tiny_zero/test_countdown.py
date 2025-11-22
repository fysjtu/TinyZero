# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
测试 countdown.py 中的奖励评分函数
"""
import sys
import os

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from verl.utils.reward_score.countdown import (
    extract_solution,
    validate_equation,
    evaluate_equation,
    compute_score
)


def test_extract_solution():
    """测试 extract_solution 函数 - 从解决方案字符串中提取方程"""
    print("\n=== 测试 extract_solution ===")
    
    # 测试用例1: 包含 "Assistant:" 标记的标准格式
    solution1 = "User: What is 5+3?\nAssistant: The answer is <answer>5+3</answer>"
    result1 = extract_solution(solution1)
    assert result1 == "5+3", f"期望 '5+3', 得到 '{result1}'"
    print("✓ 测试1通过: 标准 Assistant: 格式")
    
    # 测试用例2: 包含 "<|im_start|>assistant" 标记的格式
    solution2 = "User: Calculate 10*2\n<|im_start|>assistant\nThe result is <answer>10*2</answer>"
    result2 = extract_solution(solution2)
    assert result2 == "10*2", f"期望 '10*2', 得到 '{result2}'"
    print("✓ 测试2通过: im_start 格式")
    
    # 测试用例3: 多个 <answer> 标签，应该取最后一个
    solution3 = "Assistant: First try <answer>5+3</answer>, then <answer>8*2</answer>"
    result3 = extract_solution(solution3)
    assert result3 == "8*2", f"期望 '8*2', 得到 '{result3}'"
    print("✓ 测试3通过: 多个答案标签取最后一个")
    
    # 测试用例4: 没有 Assistant 标记，应该返回 None
    solution4 = "Just some text <answer>5+3</answer>"
    result4 = extract_solution(solution4)
    assert result4 is None, f"期望 None, 得到 '{result4}'"
    print("✓ 测试4通过: 无 Assistant 标记返回 None")
    
    # 测试用例5: 有 Assistant 标记但没有 <answer> 标签
    solution5 = "Assistant: I don't know the answer"
    result5 = extract_solution(solution5)
    assert result5 is None, f"期望 None, 得到 '{result5}'"
    print("✓ 测试5通过: 无 answer 标签返回 None")
    
    # 测试用例6: 复杂方程
    solution6 = "Assistant: The equation is <answer>(5+3)*2-1</answer>"
    result6 = extract_solution(solution6)
    assert result6 == "(5+3)*2-1", f"期望 '(5+3)*2-1', 得到 '{result6}'"
    print("✓ 测试6通过: 复杂方程提取")
    
    # 测试用例7: 多行文本，取最后一行
    solution7 = "Assistant: Line 1\nLine 2\nFinal answer: <answer>10+5</answer>"
    result7 = extract_solution(solution7)
    assert result7 == "10+5", f"期望 '10+5', 得到 '{result7}'"
    print("✓ 测试7通过: 多行文本取最后一行")
    
    print("所有 extract_solution 测试通过！\n")


def test_validate_equation():
    """测试 validate_equation 函数 - 验证方程使用的数字"""
    print("\n=== 测试 validate_equation ===")
    
    # 测试用例1: 正确使用所有数字，每个数字用一次
    equation1 = "5+3+2"
    numbers1 = [5, 3, 2]
    assert validate_equation(equation1, numbers1) == True, "应该验证通过"
    print("✓ 测试1通过: 正确使用所有数字")
    
    # 测试用例2: 数字顺序不同但集合相同
    equation2 = "2+5+3"
    numbers2 = [5, 3, 2]
    assert validate_equation(equation2, numbers2) == True, "应该验证通过（顺序无关）"
    print("✓ 测试2通过: 数字顺序不同但集合相同")
    
    # 测试用例3: 缺少数字
    equation3 = "5+3"
    numbers3 = [5, 3, 2]
    assert validate_equation(equation3, numbers3) == False, "应该验证失败（缺少数字）"
    print("✓ 测试3通过: 缺少数字检测")
    
    # 测试用例4: 使用了额外的数字
    equation4 = "5+3+2+1"
    numbers4 = [5, 3, 2]
    assert validate_equation(equation4, numbers4) == False, "应该验证失败（额外数字）"
    print("✓ 测试4通过: 额外数字检测")
    
    # 测试用例5: 复杂表达式
    equation5 = "(5+3)*2"
    numbers5 = [5, 3, 2]
    assert validate_equation(equation5, numbers5) == True, "应该验证通过"
    print("✓ 测试5通过: 复杂表达式验证")
    
    # 测试用例6: 包含除法的表达式
    equation6 = "10/2+3"
    numbers6 = [10, 2, 3]
    assert validate_equation(equation6, numbers6) == True, "应该验证通过"
    print("✓ 测试6通过: 除法表达式验证")
    
    # 测试用例7: 空方程
    equation7 = ""
    numbers7 = [5, 3, 2]
    assert validate_equation(equation7, numbers7) == False, "应该验证失败"
    print("✓ 测试7通过: 空方程检测")
    
    # 测试用例8: 无效的方程字符串（会导致异常）
    equation8 = "abc+def"
    numbers8 = [5, 3, 2]
    assert validate_equation(equation8, numbers8) == False, "应该验证失败（异常处理）"
    print("✓ 测试8通过: 异常处理")
    
    print("所有 validate_equation 测试通过！\n")


def test_evaluate_equation():
    """测试 evaluate_equation 函数 - 安全评估方程"""
    print("\n=== 测试 evaluate_equation ===")
    
    # 测试用例1: 简单加法
    equation1 = "5+3"
    result1 = evaluate_equation(equation1)
    assert result1 == 8, f"期望 8, 得到 {result1}"
    print(f"✓ 测试1通过: 简单加法 {equation1} = {result1}")
    
    # 测试用例2: 简单乘法
    equation2 = "5*3"
    result2 = evaluate_equation(equation2)
    assert result2 == 15, f"期望 15, 得到 {result2}"
    print(f"✓ 测试2通过: 简单乘法 {equation2} = {result2}")
    
    # 测试用例3: 带括号的表达式
    equation3 = "(5+3)*2"
    result3 = evaluate_equation(equation3)
    assert result3 == 16, f"期望 16, 得到 {result3}"
    print(f"✓ 测试3通过: 带括号表达式 {equation3} = {result3}")
    
    # 测试用例4: 除法
    equation4 = "10/2"
    result4 = evaluate_equation(equation4)
    assert result4 == 5.0, f"期望 5.0, 得到 {result4}"
    print(f"✓ 测试4通过: 除法 {equation4} = {result4}")
    
    # 测试用例5: 减法
    equation5 = "10-3"
    result5 = evaluate_equation(equation5)
    assert result5 == 7, f"期望 7, 得到 {result5}"
    print(f"✓ 测试5通过: 减法 {equation5} = {result5}")
    
    # 测试用例6: 复杂表达式
    equation6 = "(10+5)*2-3"
    result6 = evaluate_equation(equation6)
    assert result6 == 27, f"期望 27, 得到 {result6}"
    print(f"✓ 测试6通过: 复杂表达式 {equation6} = {result6}")
    
    # 测试用例7: 包含空格
    equation7 = "5 + 3 * 2"
    result7 = evaluate_equation(equation7)
    assert result7 == 11, f"期望 11, 得到 {result7}"  # 注意运算符优先级
    print(f"✓ 测试7通过: 包含空格 {equation7} = {result7}")
    
    # 测试用例8: 无效字符（应该返回 None）
    equation8 = "5+3+abc"
    result8 = evaluate_equation(equation8)
    assert result8 is None, f"期望 None, 得到 {result8}"
    print("✓ 测试8通过: 无效字符检测")
    
    # 测试用例9: 空字符串（应该返回 None）
    equation9 = ""
    result9 = evaluate_equation(equation9)
    assert result9 is None, f"期望 None, 得到 {result9}"
    print("✓ 测试9通过: 空字符串检测")
    
    # 测试用例10: 除零（应该返回 None 或抛出异常）
    equation10 = "10/0"
    result10 = evaluate_equation(equation10)
    # 除零会抛出异常，被捕获后返回 None
    assert result10 is None, f"期望 None（除零异常）, 得到 {result10}"
    print("✓ 测试10通过: 除零异常处理")
    
    print("所有 evaluate_equation 测试通过！\n")


def test_compute_score():
    """测试 compute_score 函数 - 计算奖励分数"""
    print("\n=== 测试 compute_score ===")
    
    # 测试用例1: 完全正确的答案
    solution1 = "Assistant: The answer is <answer>5+3</answer>"
    ground_truth1 = {'target': 8, 'numbers': [5, 3]}
    score1 = compute_score(solution1, ground_truth1, score=1.0, format_score=0.1)
    assert score1 == 1.0, f"期望 1.0, 得到 {score1}"
    print(f"✓ 测试1通过: 完全正确答案，得分 {score1}")
    
    # 测试用例2: 格式正确但答案错误
    solution2 = "Assistant: The answer is <answer>5+3</answer>"
    ground_truth2 = {'target': 10, 'numbers': [5, 3]}  # 目标值不同
    score2 = compute_score(solution2, ground_truth2, score=1.0, format_score=0.1)
    assert score2 == 0.1, f"期望 0.1, 得到 {score2}"
    print(f"✓ 测试2通过: 格式正确但答案错误，得分 {score2}")
    
    # 测试用例3: 使用了错误的数字
    solution3 = "Assistant: The answer is <answer>5+4</answer>"
    ground_truth3 = {'target': 9, 'numbers': [5, 3]}  # 使用了4而不是3
    score3 = compute_score(solution3, ground_truth3, score=1.0, format_score=0.1)
    assert score3 == 0.1, f"期望 0.1（格式分）, 得到 {score3}"
    print(f"✓ 测试3通过: 使用了错误数字，得分 {score3}")
    
    # 测试用例4: 没有提取到答案（返回0）
    solution4 = "Assistant: I don't know"
    ground_truth4 = {'target': 8, 'numbers': [5, 3]}
    score4 = compute_score(solution4, ground_truth4, score=1.0, format_score=0.1)
    assert score4 == 0.0, f"期望 0.0, 得到 {score4}"
    print(f"✓ 测试4通过: 无答案提取，得分 {score4}")
    
    # 测试用例5: 复杂表达式，正确答案
    solution5 = "Assistant: The equation is <answer>(5+3)*2</answer>"
    ground_truth5 = {'target': 16, 'numbers': [5, 3, 2]}
    score5 = compute_score(solution5, ground_truth5, score=1.0, format_score=0.1)
    assert score5 == 1.0, f"期望 1.0, 得到 {score5}"
    print(f"✓ 测试5通过: 复杂表达式正确答案，得分 {score5}")
    
    # 测试用例6: 浮点数精度测试
    solution6 = "Assistant: The answer is <answer>10/3</answer>"
    ground_truth6 = {'target': 10/3, 'numbers': [10, 3]}
    score6 = compute_score(solution6, ground_truth6, score=1.0, format_score=0.1)
    assert score6 == 1.0, f"期望 1.0（浮点精度容差）, 得到 {score6}"
    print(f"✓ 测试6通过: 浮点数精度测试，得分 {score6}")
    
    # 测试用例7: 使用 im_start 格式
    solution7 = "<|im_start|>assistant\nThe answer is <answer>5+3</answer>"
    ground_truth7 = {'target': 8, 'numbers': [5, 3]}
    score7 = compute_score(solution7, ground_truth7, score=1.0, format_score=0.1)
    assert score7 == 1.0, f"期望 1.0, 得到 {score7}"
    print(f"✓ 测试7通过: im_start 格式，得分 {score7}")
    
    # 测试用例8: 自定义分数参数
    solution8 = "Assistant: The answer is <answer>5+3</answer>"
    ground_truth8 = {'target': 8, 'numbers': [5, 3]}
    score8 = compute_score(solution8, ground_truth8, score=2.0, format_score=0.5)
    assert score8 == 2.0, f"期望 2.0, 得到 {score8}"
    print(f"✓ 测试8通过: 自定义分数参数，得分 {score8}")
    
    # 测试用例9: 无效方程（无法评估）
    solution9 = "Assistant: The answer is <answer>5+abc</answer>"
    ground_truth9 = {'target': 8, 'numbers': [5, 3]}
    score9 = compute_score(solution9, ground_truth9, score=1.0, format_score=0.1)
    assert score9 == 0.1, f"期望 0.1（格式分）, 得到 {score9}"
    print(f"✓ 测试9通过: 无效方程，得分 {score9}")
    
    print("所有 compute_score 测试通过！\n")


def test_edge_cases():
    """测试边界情况和特殊情况"""
    print("\n=== 测试边界情况 ===")
    
    # 边界用例1: 单个数字
    equation1 = "5"
    numbers1 = [5]
    assert validate_equation(equation1, numbers1) == True, "单个数字应该验证通过"
    print("✓ 边界测试1通过: 单个数字")
    
    # 边界用例2: 大数字
    equation2 = "1000+500"
    numbers2 = [1000, 500]
    assert validate_equation(equation2, numbers2) == True, "大数字应该验证通过"
    result2 = evaluate_equation(equation2)
    assert result2 == 1500, f"期望 1500, 得到 {result2}"
    print("✓ 边界测试2通过: 大数字")
    
    # 边界用例3: 负数（虽然 countdown 任务通常不用负数）
    equation3 = "5-10"
    result3 = evaluate_equation(equation3)
    assert result3 == -5, f"期望 -5, 得到 {result3}"
    print("✓ 边界测试3通过: 负数结果")
    
    # 边界用例4: 多层括号
    equation4 = "((5+3)*2)+1"
    numbers4 = [5, 3, 2, 1]
    assert validate_equation(equation4, numbers4) == True, "多层括号应该验证通过"
    result4 = evaluate_equation(equation4)
    assert result4 == 17, f"期望 17, 得到 {result4}"
    print("✓ 边界测试4通过: 多层括号")
    
    # 边界用例5: 空 ground_truth
    solution5 = "Assistant: <answer>5+3</answer>"
    ground_truth5 = {'target': 8, 'numbers': []}
    score5 = compute_score(solution5, ground_truth5, score=1.0, format_score=0.1)
    assert score5 == 0.1, f"期望 0.1（空数字列表）, 得到 {score5}"
    print("✓ 边界测试5通过: 空数字列表")
    
    print("所有边界情况测试通过！\n")


if __name__ == "__main__":
    """主测试入口 - 运行所有测试用例"""
    print("=" * 60)
    print("开始测试 countdown.py 模块")
    print("=" * 60)
    
    try:
        # 运行所有测试函数
        test_extract_solution()
        test_validate_equation()
        test_evaluate_equation()
        test_compute_score()
        test_edge_cases()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

