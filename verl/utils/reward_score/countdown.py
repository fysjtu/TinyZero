"""
Countdown 任务的奖励评分模块。

该模块实现了用于评估 Countdown 游戏任务（算术拼图游戏）的奖励函数。
Countdown 游戏的规则是：给定一组数字和一个目标值，使用这些数字（每个数字只能用一次）
通过加、减、乘、除运算得到目标值。

主要功能：
1. extract_solution: 从模型生成的文本中提取算术方程
2. validate_equation: 验证方程是否使用了正确的数字
3. evaluate_equation: 安全地计算算术表达式的值
4. compute_score: 综合评分函数，用于强化学习训练中的奖励计算

使用场景：
- 强化学习（RL）训练中的奖励函数
- 模型生成结果的自动评估
- Countdown 游戏任务的验证
"""
import re
import random
import ast
import operator


def extract_solution(solution_str):
    """
    从模型生成的解决方案字符串中提取算术方程。
    
    该函数用于从包含对话格式的文本中提取被 <answer> 标签包裹的算术表达式。
    支持两种常见的对话格式标记："Assistant:" 和 "<|im_start|>assistant"。
    
    Args:
        solution_str (str): 包含模型生成解决方案的完整字符串，可能包含对话标记和答案标签
        
    Returns:
        str or None: 提取出的算术方程字符串（去除首尾空格），如果未找到则返回 None
        
    Examples:
        >>> extract_solution("Assistant: The answer is <answer>5+3</answer>")
        '5+3'
        >>> extract_solution("<|im_start|>assistant\n<answer>(5+3)*2</answer>")
        '(5+3)*2'
        >>> extract_solution("No answer here")
        None
    """
    # 步骤1: 定位 Assistant 回复部分，移除标记之前的所有内容
    # 优先查找 "Assistant:" 标记（常见于指令微调模型的输出格式）
    if "Assistant:" in solution_str:
        # 使用 split 分割，limit=1 确保只分割一次，取标记后的内容
        solution_str = solution_str.split("Assistant:", 1)[1]
    # 如果没找到，尝试查找 "<|im_start|>assistant" 标记（ChatML 格式）
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # 如果两种标记都不存在，说明格式不符合预期，返回 None
        return None
    
    # 步骤2: 提取最后一行内容（通常答案在最后一行）
    # 这样可以处理多行回复的情况，只关注最后一行
    solution_str = solution_str.split('\n')[-1]

    # 步骤3: 使用正则表达式提取 <answer> 标签内的内容
    # 使用非贪婪匹配 (.*?) 以匹配第一个 </answer> 标签
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    
    if matches:
        # 如果找到多个匹配，取最后一个（处理模型可能多次输出答案的情况）
        # group(1) 获取第一个捕获组的内容，strip() 去除首尾空格
        final_answer = matches[-1].group(1).strip()
    else:
        # 如果没有找到 <answer> 标签，返回 None
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """
    验证方程是否只使用了给定的数字，且每个数字恰好使用一次。
    
    该函数用于确保模型生成的方程符合 Countdown 游戏的规则：
    1. 只能使用给定的数字列表中的数字
    2. 每个数字必须使用且只能使用一次
    
    Args:
        equation_str (str): 待验证的算术方程字符串，例如 "5+3*2"
        available_numbers (list): 可用的数字列表，例如 [5, 3, 2]
        
    Returns:
        bool: 如果方程符合规则返回 True，否则返回 False
        
    Examples:
        >>> validate_equation("5+3", [5, 3])
        True
        >>> validate_equation("5+3+2", [5, 3])  # 使用了额外的数字 2
        False
        >>> validate_equation("5+3", [5, 3, 2])  # 缺少数字 2
        False
        >>> validate_equation("(5+3)*2", [5, 3, 2])
        True
    """
    try:
        # 步骤1: 从方程字符串中提取所有数字
        # 使用正则表达式 \d+ 匹配所有连续的数字（整数）
        # 将匹配结果转换为整数列表
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # 步骤2: 对两个数字列表进行排序以便比较
        # 排序后可以忽略数字在方程中的顺序，只关注数字集合是否相同
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # 步骤3: 比较排序后的列表是否完全一致
        # 如果一致，说明方程使用了所有可用数字且每个数字只用一次
        return numbers_in_eq == available_numbers
    except:
        # 如果过程中出现任何异常（如方程格式错误、无法解析等），返回 False
        return False


def evaluate_equation(equation_str):
    """
    安全地计算算术表达式的值。
    
    该函数使用 eval() 计算算术表达式，但通过字符白名单和受限执行环境来降低安全风险。
    只允许数字、基本运算符（+、-、*、/）、括号和空格。
    
    Args:
        equation_str (str): 待计算的算术表达式字符串，例如 "5+3*2" 或 "(10+5)/3"
        
    Returns:
        float or int or None: 计算结果，如果表达式无效或计算失败则返回 None
        
    Examples:
        >>> evaluate_equation("5+3")
        8
        >>> evaluate_equation("(5+3)*2")
        16
        >>> evaluate_equation("10/2")
        5.0
        >>> evaluate_equation("5+abc")  # 包含非法字符
        None
        >>> evaluate_equation("10/0")  # 除零错误
        None
    """
    try:
        # 步骤1: 使用正则表达式进行字符白名单验证
        # 只允许：数字(\d)、加号(+)、减号(-)、乘号(*)、除号(/)、括号(())、点号(.)、空格(\s)
        # 注意：在字符类 [] 中，+ 不需要转义，但为了清晰可以转义
        # ^ 和 $ 确保匹配整个字符串，防止部分匹配
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            # 如果包含非法字符，抛出异常
            raise ValueError("Invalid characters in equation.")

        # 步骤2: 在受限环境中执行表达式
        # eval() 的第二个参数限制全局命名空间，第三个参数限制局部命名空间
        # {"__builtins__": None} 禁用所有内置函数，防止代码注入攻击
        # {} 空的局部命名空间，进一步限制可执行的代码
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        # 捕获所有异常（包括除零、语法错误、非法字符等），返回 None 表示计算失败
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """
    Countdown 任务的评分函数。
    
    该函数是奖励模型的核心，用于评估模型生成的解决方案是否正确。
    评分流程分为三个阶段：
    1. 提取阶段：从解决方案文本中提取算术方程
    2. 验证阶段：检查方程是否使用了正确的数字
    3. 评估阶段：计算方程结果并与目标值比较
    
    评分规则：
    - 完全正确（格式正确 + 数字正确 + 结果正确）：返回 score（默认 1.0）
    - 格式正确但答案错误（格式正确 + 数字正确但结果错误）：返回 format_score（默认 0.1）
    - 格式错误或无法提取：返回 0
    
    Args:
        solution_str (str): 模型生成的完整解决方案文本，可能包含对话格式和答案标签
        ground_truth (dict): 包含正确答案信息的字典，必须包含以下键：
            - 'target' (int/float): 目标数值，方程应该计算得到的结果
            - 'numbers' (list): 可用的数字列表，方程必须使用这些数字且每个只用一次
        method (str): 提取方法（当前未使用，保留用于未来扩展），默认 'strict'
        format_score (float): 格式正确但答案错误时的分数，默认 0.1
        score (float): 完全正确时的分数，默认 1.0
        
    Returns:
        float: 评分结果，范围在 [0, score] 之间
        
    Examples:
        >>> ground_truth = {'target': 8, 'numbers': [5, 3]}
        >>> compute_score("Assistant: <answer>5+3</answer>", ground_truth)
        1.0  # 完全正确
        >>> compute_score("Assistant: <answer>5+3</answer>", {'target': 10, 'numbers': [5, 3]})
        0.1  # 格式正确但结果错误
        >>> compute_score("Assistant: <answer>5+4</answer>", {'target': 9, 'numbers': [5, 3]})
        0.1  # 使用了错误的数字
        >>> compute_score("Assistant: I don't know", ground_truth)
        0.0  # 无法提取答案
    """
    # 从 ground_truth 中提取目标值和可用数字列表
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    # 步骤1: 从解决方案文本中提取算术方程
    equation = extract_solution(solution_str=solution_str)
    
    # 随机决定是否打印调试信息（1/64 的概率，减少日志输出量）
    # 用于调试和监控，不影响评分逻辑
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    # 步骤2: 检查是否成功提取到方程
    if equation is None:
        # 如果无法提取方程（格式错误或没有答案标签），返回 0 分
        if do_print:
            print(f"No equation found")
        return 0
    
    # 步骤3: 验证方程是否使用了正确的数字
    # 检查方程是否只使用了给定的数字，且每个数字恰好使用一次
    if not validate_equation(equation, numbers):
        # 如果数字验证失败，返回格式分数（部分奖励，鼓励正确的格式）
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # 步骤4: 计算方程的值并与目标值比较
    try:
        # 安全地计算方程的值
        result = evaluate_equation(equation)
        
        # 如果计算失败（表达式无效、除零等），返回格式分数
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        # 比较计算结果与目标值
        # 使用 1e-5 的容差来处理浮点数精度问题（例如 10/3 的结果）
        if abs(result - target) < 1e-5:  # Account for floating point precision
            # 结果完全正确，返回满分
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            # 结果不正确，但格式和数字都正确，返回格式分数
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        # 如果评估过程中出现任何未预期的异常，返回格式分数
        if do_print:
            print(f"Error evaluating equation")
        return format_score 