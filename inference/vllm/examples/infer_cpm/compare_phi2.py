import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

prompts = [
    # "input: 请告诉我把大象放进冰箱需要几步？output:",
    # "input:请给我写一首描绘极寒的冬天的古诗 output:",
    # "input:周杰伦都写了哪些歌呢?尽可能多地列举一些 output:",
    # "input:铁做氧化还原反应的化学方程式是什么 output:",
    # "input:linux里 watch -n1 file.txt命令是什么意思 output:",
    # "input:帮我计算一下 12 + 13 - 2 * 5 = ? 请一步一步来 output:",
    # "input:这段python代码 ```def add_number_minus1(x, y):\n    return x + y - 1\n``` 用C语言应该怎么写？output:",
    # "input:假唱是指什么？请剖析五月天假唱的正面影响和负面影响, 他们应该假唱吗? output:",
    "Q:Which songs has Jay Chou written? Please list as many as possible. A:",
    "Q:What is the chemical equation for the redox reaction of iron? A:",
    "Q:Tell me a joke about a classmate who wanted to go to the bathroom during class. A:",
    "Q:What does lip-syncing mean? Please analyze the positive and negative impacts of the band Mayday lip-syncing. Should they lip-sync? A:"
]

prompts = [
    "<用户>Write five words that start with “en”, then write the result of “77+33”<AI>",
    "<用户>Here we have a book, 9 eggs, a laptop, a bottle and a nail. Please tell me how to stack them onto each other in a stable manner.<AI>",
    "<用户>A group of students are planning to go on a field trip to a museum. They need to decide how many buses to rent and how to divide the students among the buses. Each bus can hold up to 40 students, but the museum can only accommodate 120 students at a time. The group has a budget of $800 for the bus rental, and each bus costs $200 per day. How many buses should the group rent, and how many students should go on each bus? Explain your reasoning.<AI>",
    """<用户>Selena, Jennifer and Miley wear a blue dress, yellow dress, and green dress in an unknown order. It is known that:

1) If Selena wears blue, then Jennifer wears green.
2) If Selena wears yellow, then Miley wears green.
3) If Jennifer does not wear yellow, then Miley wears blue.

What is the color of the dress Selena is wearing?<AI>""",
    """<用户>Given the following premise:
It is given that Amy, Bernadette and Penny are good friends of Sheldon and Leonard.
Leslie is a good friend of Leonard.

We can deduce that Leslie is a friend of Sheldon as well.

Does this deduction follow the premise? Explain your reasoning.<AI>""",
    """<用户>A group of five friends are going to watch a movie together. They have to choose between three genres: comedy, horror, and action. Each friend has a different preference for the genre. Here are some clues to help you figure out their preferences: Alice likes comedy more than horror, but less than action. Bob likes horror more than comedy, but less than action. Carol likes action more than horror, but less than comedy. David likes comedy more than action, but less than horror. Eve likes horror more than action, but less than comedy. What is the order of preference for each friend from most to least liked genre? Write your answer using the following format: Friend: Genre > Genre > Genre <AI>""",
    "<用户>If you were in a race and passed the person in second place, what place would you be in now?<AI>",
    "<用户>Which one is more south? California or New York?<AI>",
    "<用户>linux里 watch -n1 file.txt命令是什么意思<AI>",
    "<用户>Translate this sentence into Russian: '如何用Python创建一个简单的网页爬虫？'.<AI>",
    """<用户>Translate this sentence into French: "I am a fresh man on Chinese, do you know how this sentence is translated: 如何用Python创建一个简单的网页爬虫？" <AI>""",
    "<用户>Micro-expressions mean that people express their inner feelings to each other by making some expressions.Between different expressions made by people or in a certain expression, the face will \"leak\" other information.The shortest-lasting micro-expressions can last 1 / 25 seconds, although a subconscious expression may only last for a moment, it is easy to expose emotions.When the face is making an expression, these extremely short-term expressions suddenly flash by, and sometimes the opposite mood.\nAccording to the above definition, which of the following is a micro-expression?\nA.After Wang was frightened, his face continued to twitch\nB.The spy sends a message to associates in the blink of an eye\nC.The sales clerk may flash a contemptuous smirk when he smiles in front of a shabby customer.\nD.Walking against the biting cold wind, Xiao Li's upper and lower teeth kept shaking and colliding\nA:<AI>",
    "<用户>A’s brother was half her age when she was 6 how old is her brother when she’s 42?<AI>",
] 

for sent in prompts:
    inputs = tokenizer(sent, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print("-"*20)
    print(text)
    print("="*20)