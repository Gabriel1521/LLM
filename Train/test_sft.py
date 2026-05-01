import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

model_dir="C:/Users/Gabriela/.cache/modelscope/hub/models/qwen/Qwen1___5-0___5B-Chat"

# 4bit 量化（和训练时一致）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

# ======================
# 2. 加载你训练好的 LoRA 权重
# ======================
model = PeftModel.from_pretrained(base_model, "Train/qwen-gsm8k-lora-final")

# ======================
# 3. 推理函数（输入数学题，输出答案）
# ======================
def generate_answer(question):
    # 构造输入（和训练格式一样）
    prompt = f"用户：{question}\n助手："
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,   # 数学题足够长
            temperature=0.1,      # 输出更稳定
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("助手：")[-1].strip()

# ======================
# 4. 测试！
# ======================
if __name__ == "__main__":
    # 你可以随便换题目
    question1 = "小明有5个苹果，再买3个，一共有多少个？"
    question2 = "What is 5 + 3?"
    question3 = "If x + 5 = 10, what is x?"

    print("问题1：", question1)
    print("回答：", generate_answer(question1))
    print("-"*50)

    print("问题2：", question2)
    print("回答：", generate_answer(question2))
    print("-"*50)

    print("问题3：", question3)
    print("回答：", generate_answer(question3))