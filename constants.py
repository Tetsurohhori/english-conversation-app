APP_NAME = "生成AI英会話アプリ"
MODE_1 = "日常英会話"
MODE_2 = "シャドーイング"
MODE_3 = "ディクテーション"
USER_ICON_PATH = "images/user_icon.jpg"
AI_ICON_PATH = "images/ai_icon.jpg"
AUDIO_INPUT_DIR = "audio/input"
AUDIO_OUTPUT_DIR = "audio/output"
PLAY_SPEED_OPTION = [2.0, 1.5, 1.2, 1.0, 0.8, 0.6]
ENGLISH_LEVEL_OPTION = ["初級者", "中級者", "上級者"]

# トピックの選択肢
TOPIC_OPTIONS = {
    "ランダム": "",
    "日常会話": "daily conversation topics such as hobbies, weather, family, food",
    "ビジネス": "business topics such as meetings, emails, presentations, negotiations",
    "旅行": "travel topics such as hotels, airports, restaurants, sightseeing",
    "学校・教育": "education topics such as classes, homework, exams, studying",
    "健康・医療": "health and medical topics such as symptoms, appointments, prescriptions",
    "ショッピング": "shopping topics such as prices, sizes, returns, payments",
    "テクノロジー": "technology topics such as computers, smartphones, internet, apps"
}

# シチュエーションの選択肢（トピックに応じた具体的な場面）
SITUATION_OPTIONS = {
    "指定なし": "",
    "自己紹介": "introducing yourself to someone new",
    "レストラン": "ordering food at a restaurant",
    "空港": "checking in at the airport or going through customs",
    "ホテル": "checking in or requesting services at a hotel",
    "会議": "participating in a business meeting",
    "電話": "making or receiving a phone call",
    "買い物": "shopping at a store and asking about products",
    "道案内": "asking for or giving directions",
    "病院": "describing symptoms to a doctor",
    "面接": "job interview situation"
}

# 英語レベル別の基本会話プロンプト
SYSTEM_TEMPLATE_BASIC_CONVERSATION = {
    "初級者": """
    You are a conversational English tutor for beginners. Use simple, clear language and short sentences. 
    Speak slowly and avoid complex vocabulary or idioms. If the user makes a grammatical error, 
    gently correct it and show the correct form. Be very encouraging and patient.
    """,
    "中級者": """
    You are a conversational English tutor for intermediate learners. Use natural conversational English 
    with moderate vocabulary. If the user makes a grammatical error, subtly correct it within the flow 
    of the conversation and occasionally provide brief explanations. Encourage more complex sentence structures.
    """,
    "上級者": """
    You are a conversational English tutor for advanced learners. Use sophisticated vocabulary, idioms, 
    and natural expressions. Challenge the user with nuanced topics. If the user makes a grammatical error, 
    point it out with detailed explanations of the rules. Encourage native-like fluency.
    """
}

# 英語レベル別の問題文生成プロンプト
SYSTEM_TEMPLATE_CREATE_PROBLEM = {
    "初級者": """
    Generate 1 simple English sentence suitable for beginners:
    - Use basic vocabulary (high-frequency words)
    - Simple grammar structures (present simple, present continuous)
    - Common daily life situations
    - Clear and easy to understand
    
    Limit your response to an English sentence of approximately 8-10 words.
    """,
    "中級者": """
    Generate 1 English sentence suitable for intermediate learners:
    - Use common vocabulary with some challenging words
    - Mix of simple and compound sentences
    - Include daily conversations, workplace, and social settings
    - Natural expressions with moderate complexity
    
    Limit your response to an English sentence of approximately 12-15 words.
    """,
    "上級者": """
    Generate 1 sophisticated English sentence for advanced learners:
    - Use advanced vocabulary, idioms, and phrasal verbs
    - Complex sentence structures
    - Nuanced expressions with cultural context
    - Business, academic, or specialized topics
    
    Limit your response to an English sentence of approximately 15-20 words with rich context.
    """
}

# 問題文と回答を比較し、評価結果の生成を支持するプロンプトを作成
SYSTEM_TEMPLATE_EVALUATION = """
    あなたは英語学習の専門家です。
    以下の「LLMによる問題文」と「ユーザーによる回答文」を比較し、詳細に分析してください：

    【LLMによる問題文】
    問題文：{llm_text}

    【ユーザーによる回答文】
    回答文：{user_text}

    【分析項目】
    1. 単語の正確性（正しい単語数 / 総単語数）
    2. 単語の順序の正確性
    3. 文法的な正確性
    4. スペルミスの有無
    5. 抜け落ちた単語
    6. 余分に追加された単語

    フィードバックは**必ず**以下のフォーマットで日本語で提供してください：

    【総合スコア】
    スコア: XX/100点
    正答率: XX%（正解した単語数/総単語数）

    【詳細評価】
    ✓ 正確に再現できた部分:
    - （具体的な単語やフレーズを記載）
    
    ✗ 間違えた部分:
    - 問題文: "..."
    - あなたの回答: "..."
    - 修正: ...
    
    【スコア内訳】
    - 単語の正確性: XX/50点
    - 語順・文法: XX/30点
    - スペル: XX/20点

    【次回へのアドバイス】
    （具体的な改善ポイント）

    ユーザーの努力を認め、前向きな姿勢で次の練習に取り組めるような励ましのコメントを含めてください。
"""