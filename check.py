from wordseg.solution import Solution

solution = Solution()

funcs = {
    "CRF": solution.crf_predict,
    "BiLSTM-CRF": solution.dnn_predict
}

inputs = ["我爱北京天安门", "今天天气怎么样", "中华人民共和国","難以考證的漢字"]

for f_name, func in funcs.items():
    outputs = func(inputs)
    print(f"对于输入{inputs}，你的输出是{outputs}。")


for f_name, func in funcs.items():
    print(f"\n{f_name}模型 -->")
    examples = open("/Users/mengtong/Desktop/lab2_test_data/input.utf8", encoding="utf8").readlines()
    examples = [ele.strip() for ele in examples]
    gold = open("/Users/mengtong/Desktop/lab2_test_data/gold.utf8", encoding="utf8").readlines()
    gold = [ele.strip() for ele in gold]
    pred = func(examples)  # 你预测的结果。
    accuracys = []
    if pred is not None:
        for i in range(len(examples)):
            corr = [1 if a == b else 0 for a, b in zip(str(gold[i]), str(pred[i]))]
            accu = sum(corr) / len(corr)
            accuracys.append(accu)
        print(f"准确率：{sum(accuracys) / len(accuracys)}")
    else:
        print(f"{f_name}暂无结果")
