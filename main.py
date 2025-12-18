import streamlit as st
import os
import time
from time import sleep
from pathlib import Path
from streamlit.components.v1 import html
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import functions as ft
import constants as ct


# 各種設定
load_dotenv()
st.set_page_config(
    page_title=ct.APP_NAME
)

# タイトル表示
st.markdown(f"## {ct.APP_NAME}")

# 初期処理
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.start_flg = False
    st.session_state.pause_flg = False  # 一時中断フラグ
    st.session_state.pre_mode = ""
    st.session_state.shadowing_flg = False
    st.session_state.shadowing_button_flg = False
    st.session_state.shadowing_count = 0
    st.session_state.shadowing_first_flg = True
    st.session_state.shadowing_audio_input_flg = False
    st.session_state.shadowing_evaluation_first_flg = True
    st.session_state.dictation_flg = False
    st.session_state.dictation_button_flg = False
    st.session_state.dictation_count = 0
    st.session_state.dictation_first_flg = True
    st.session_state.dictation_chat_message = ""
    st.session_state.dictation_evaluation_first_flg = True
    st.session_state.chat_open_flg = False
    st.session_state.problem = ""
    
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        max_token_limit=1000,
        return_messages=True
    )

    # 英語レベル別のChainは動的に作成（初期化時には作成しない）
    st.session_state.chain_basic_conversation = None

# 最上部の設定エリア（再生速度・モード・英語レベル）
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.session_state.speed = st.selectbox(label="再生速度", options=ct.PLAY_SPEED_OPTION, index=3)
with col2:
    st.session_state.mode = st.selectbox(label="モード", options=[ct.MODE_1, ct.MODE_2, ct.MODE_3])
    # モードを変更した際の処理
    if st.session_state.mode != st.session_state.pre_mode:
        # 自動でそのモードの処理が実行されないようにする
        st.session_state.start_flg = False
        # 「日常英会話」選択時の初期化処理
        if st.session_state.mode == ct.MODE_1:
            st.session_state.dictation_flg = False
            st.session_state.shadowing_flg = False
        # 「シャドーイング」選択時の初期化処理
        st.session_state.shadowing_count = 0
        if st.session_state.mode == ct.MODE_2:
            st.session_state.dictation_flg = False
        # 「ディクテーション」選択時の初期化処理
        st.session_state.dictation_count = 0
        if st.session_state.mode == ct.MODE_3:
            st.session_state.shadowing_flg = False
        # チャット入力欄を非表示にする
        st.session_state.chat_open_flg = False
    st.session_state.pre_mode = st.session_state.mode
with col3:
    st.session_state.englv = st.selectbox(label="英語レベル", options=ct.ENGLISH_LEVEL_OPTION)

# トピックとシチュエーションの選択（シャドーイング・ディクテーションモードの場合のみ表示）
if st.session_state.mode in [ct.MODE_2, ct.MODE_3]:
    col5, col6 = st.columns(2)
    with col5:
        st.session_state.topic = st.selectbox(
            label="テーマ",
            options=list(ct.TOPIC_OPTIONS.keys()),
            key="topic_select"
        )
    with col6:
        st.session_state.situation = st.selectbox(
            label="場面",
            options=list(ct.SITUATION_OPTIONS.keys()),
            key="situation_select"
        )

# サイドバーに学習統計を表示
with st.sidebar:
    st.markdown("## 📊 学習統計")
    
    if 'score_history' in st.session_state and len(st.session_state.score_history) > 0:
        scores = [item['score'] for item in st.session_state.score_history]
        
        # 統計情報
        st.metric("総学習回数", len(scores))
        st.metric("平均スコア", f"{sum(scores) / len(scores):.1f}点")
        st.metric("最高スコア", f"{max(scores)}点")
        st.metric("最新スコア", f"{scores[-1]}点")
        
        # スコアの推移グラフ
        st.markdown("### スコア推移")
        import pandas as pd
        df_scores = pd.DataFrame({
            '回数': range(1, len(scores) + 1),
            'スコア': scores
        })
        st.line_chart(df_scores.set_index('回数'))
        
        # モード別の統計
        st.markdown("### モード別統計")
        mode_stats = {}
        for item in st.session_state.score_history:
            mode = item['mode']
            if mode not in mode_stats:
                mode_stats[mode] = []
            mode_stats[mode].append(item['score'])
        
        for mode, mode_scores in mode_stats.items():
            st.markdown(f"**{mode}**: 平均 {sum(mode_scores) / len(mode_scores):.1f}点 ({len(mode_scores)}回)")
        
        # リセットボタン
        if st.button("📝 学習履歴をリセット"):
            st.session_state.score_history = []
            st.rerun()
    else:
        st.info("まだ学習記録がありません。\n練習を始めましょう！")
    
    st.divider()
    st.markdown("## ℹ️ 操作説明")
    st.markdown("""
    - モードと英語レベルを選択
    - 「開始」ボタンで練習開始
    - 音声入力は「発話開始」→話す→「発話終了」
    - 5秒沈黙で自動確定
    - 「一時中断」でいつでも中断可能
    """)

with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
    st.markdown("こちらは生成AIによる音声英会話の練習アプリです。何度も繰り返し練習し、英語力をアップさせましょう。")
    st.markdown("**【操作説明】**")
    st.success("""
    - モードと英語レベルを選択し、「開始」ボタンを押して練習を始めましょう。
    - モードは「日常英会話」「シャドーイング」「ディクテーション」から選べます。
    - 音声入力：「発話開始」ボタンを押して話し、「発話終了」ボタンで確定します（5秒沈黙で自動確定）。
    - 「一時中断」ボタンを押すことで、いつでも練習を中断できます。
    """)

# 開始・一時中断ボタン（5:5配置）
col_btn1, col_btn2 = st.columns([5, 5])
with col_btn1:
    if st.session_state.start_flg:
        # 開始状態では「開始」ボタンを無効化
        st.button("開始", use_container_width=True, type="primary", disabled=True)
    else:
        st.session_state.start_flg = st.button("開始", use_container_width=True, type="primary")
with col_btn2:
    # 一時中断ボタン（開始状態の時のみ有効）
    if st.session_state.start_flg:
        if st.button("一時中断", use_container_width=True, type="secondary"):
            # 中断処理: 状態をリセット
            st.session_state.start_flg = False
            st.session_state.shadowing_flg = False
            st.session_state.dictation_flg = False
            st.session_state.chat_open_flg = False
            st.info("✋ 一時中断しました。「開始」ボタンで再開できます。")
            st.rerun()
    else:
        st.button("一時中断", use_container_width=True, type="secondary", disabled=True)

st.divider()

# メッセージリストの一覧表示
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="images/ai_icon.jpg"):
            st.markdown(message["content"])
    elif message["role"] == "user":
        with st.chat_message(message["role"], avatar="images/user_icon.jpg"):
            st.markdown(message["content"])
    else:
        st.divider()

# LLMレスポンスの下部にモード実行のボタン表示
if st.session_state.shadowing_flg:
    st.session_state.shadowing_button_flg = st.button("シャドーイング開始")
if st.session_state.dictation_flg:
    st.session_state.dictation_button_flg = st.button("ディクテーション開始")

# 「ディクテーション」モードのチャット入力受付時に実行
if st.session_state.chat_open_flg:
    st.info("AIが読み上げた音声を、画面下部のチャット欄からそのまま入力・送信してください。")

st.session_state.dictation_chat_message = st.chat_input("※「ディクテーション」選択時以外は送信不可")

if st.session_state.dictation_chat_message and not st.session_state.chat_open_flg:
    st.stop()

# 「英会話開始」ボタンが押された場合の処理
if st.session_state.start_flg:

    # モード：「ディクテーション」
    # 「ディクテーション」ボタン押下時か、「英会話開始」ボタン押下時か、チャット送信時
    if st.session_state.mode == ct.MODE_3 and (st.session_state.dictation_button_flg or st.session_state.dictation_count == 0 or st.session_state.dictation_chat_message):
        if st.session_state.dictation_first_flg:
            # 英語レベルに応じた問題文生成Chainを作成（トピック・シチュエーション考慮）
            base_prompt = ct.SYSTEM_TEMPLATE_CREATE_PROBLEM[st.session_state.englv]
            topic = st.session_state.get('topic', 'ランダム')
            situation = st.session_state.get('situation', '指定なし')
            modified_prompt = ft.create_problem_prompt_with_context(base_prompt, topic, situation)
            st.session_state.chain_create_problem = ft.create_chain(modified_prompt)
            st.session_state.dictation_first_flg = False
        # チャット入力以外
        if not st.session_state.chat_open_flg:
            with st.spinner('問題文生成中...'):
                st.session_state.problem, llm_response_audio = ft.create_problem_and_play_audio()

            st.session_state.chat_open_flg = True
            st.session_state.dictation_flg = False
            st.rerun()
        # チャット入力時の処理
        else:
            # チャット欄から入力された場合にのみ評価処理が実行されるようにする
            if not st.session_state.dictation_chat_message:
                st.stop()
            
            # AIメッセージとユーザーメッセージの画面表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.problem)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(st.session_state.dictation_chat_message)

            # LLMが生成した問題文とチャット入力値をメッセージリストに追加
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.problem})
            st.session_state.messages.append({"role": "user", "content": st.session_state.dictation_chat_message})
            
            with st.spinner('評価結果の生成中...'):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.problem,
                    user_text=st.session_state.dictation_chat_message
                )
                st.session_state.chain_evaluation = ft.create_chain(system_template)
                # 問題文と回答を比較し、評価結果の生成を指示するプロンプトを作成
                llm_response_evaluation = ft.create_evaluation()
            
            # スコアを抽出してバッジ表示
            score = ft.extract_score(llm_response_evaluation)
            if score is not None:
                ft.display_score_badge(score)
                # スコアを履歴に記録
                if 'score_history' not in st.session_state:
                    st.session_state.score_history = []
                st.session_state.score_history.append({
                    'mode': ct.MODE_3,
                    'score': score,
                    'timestamp': time.time()
                })
            
            # 評価結果のメッセージリストへの追加と表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(llm_response_evaluation)
            st.session_state.messages.append({"role": "assistant", "content": llm_response_evaluation})
            st.session_state.messages.append({"role": "other"})
            
            # 各種フラグの更新
            st.session_state.dictation_flg = True
            st.session_state.dictation_chat_message = ""
            st.session_state.dictation_count += 1
            st.session_state.chat_open_flg = False

            st.rerun()

    
    # モード：「日常英会話」
    if st.session_state.mode == ct.MODE_1:
        # 英語レベルに応じたChainを動的に作成（レベルが変更された場合に対応）
        if st.session_state.chain_basic_conversation is None or not hasattr(st.session_state, 'current_level') or st.session_state.current_level != st.session_state.englv:
            st.session_state.chain_basic_conversation = ft.create_chain(ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION[st.session_state.englv])
            st.session_state.current_level = st.session_state.englv
        
        # 音声入力を受け取って音声ファイルを作成
        audio_input_file_path = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
        ft.record_audio(audio_input_file_path)

        # 音声入力ファイルから文字起こしテキストを取得
        with st.spinner('音声入力をテキストに変換中...'):
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        # 音声入力テキストの画面表示
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        with st.spinner("回答の音声読み上げ準備中..."):
            try:
                # ユーザー入力値をLLMに渡して回答取得
                llm_response = st.session_state.chain_basic_conversation.predict(input=audio_input_text)
                
                # LLMからの回答を音声データに変換
                llm_response_audio = st.session_state.openai_obj.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=llm_response
                )

                # 一旦mp3形式で音声ファイル作成後、wav形式に変換
                audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
                ft.save_to_wav(llm_response_audio.content, audio_output_file_path)
            except Exception as e:
                st.error(f"⚠️ 回答生成中にエラーが発生しました: {str(e)}")
                st.stop()

        # 音声ファイルの読み上げ
        ft.play_wav(audio_output_file_path, speed=st.session_state.speed)

        # AIメッセージの画面表示とリストへの追加
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response)

        # ユーザー入力値とLLMからの回答をメッセージ一覧に追加
        st.session_state.messages.append({"role": "user", "content": audio_input_text})
        st.session_state.messages.append({"role": "assistant", "content": llm_response})


    # モード：「シャドーイング」
    # 「シャドーイング」ボタン押下時か、「英会話開始」ボタン押下時
    if st.session_state.mode == ct.MODE_2 and (st.session_state.shadowing_button_flg or st.session_state.shadowing_count == 0 or st.session_state.shadowing_audio_input_flg):
        if st.session_state.shadowing_first_flg:
            # 英語レベルに応じた問題文生成Chainを作成（トピック・シチュエーション考慮）
            base_prompt = ct.SYSTEM_TEMPLATE_CREATE_PROBLEM[st.session_state.englv]
            topic = st.session_state.get('topic', 'ランダム')
            situation = st.session_state.get('situation', '指定なし')
            modified_prompt = ft.create_problem_prompt_with_context(base_prompt, topic, situation)
            st.session_state.chain_create_problem = ft.create_chain(modified_prompt)
            st.session_state.shadowing_first_flg = False
        
        if not st.session_state.shadowing_audio_input_flg:
            with st.spinner('問題文生成中...'):
                st.session_state.problem, llm_response_audio = ft.create_problem_and_play_audio()

        # 音声入力を受け取って音声ファイルを作成
        st.session_state.shadowing_audio_input_flg = True
        audio_input_file_path = f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
        ft.record_audio(audio_input_file_path)
        st.session_state.shadowing_audio_input_flg = False

        with st.spinner('音声入力をテキストに変換中...'):
            # 音声入力ファイルから文字起こしテキストを取得
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        # AIメッセージとユーザーメッセージの画面表示
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(st.session_state.problem)
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)
        
        # LLMが生成した問題文と音声入力値をメッセージリストに追加
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.problem})
        st.session_state.messages.append({"role": "user", "content": audio_input_text})

        with st.spinner('評価結果の生成中...'):
            if st.session_state.shadowing_evaluation_first_flg:
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.problem,
                    user_text=audio_input_text
                )
                st.session_state.chain_evaluation = ft.create_chain(system_template)
                st.session_state.shadowing_evaluation_first_flg = False
            # 問題文と回答を比較し、評価結果の生成を指示するプロンプトを作成
            llm_response_evaluation = ft.create_evaluation()
        
        # スコアを抽出してバッジ表示
        score = ft.extract_score(llm_response_evaluation)
        if score is not None:
            ft.display_score_badge(score)
            # スコアを履歴に記録
            if 'score_history' not in st.session_state:
                st.session_state.score_history = []
            st.session_state.score_history.append({
                'mode': ct.MODE_2,
                'score': score,
                'timestamp': time.time()
            })
        
        # 評価結果のメッセージリストへの追加と表示
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response_evaluation)
        st.session_state.messages.append({"role": "assistant", "content": llm_response_evaluation})
        st.session_state.messages.append({"role": "other"})
        
        # 各種フラグの更新
        st.session_state.shadowing_flg = True
        st.session_state.shadowing_count += 1

        # 「シャドーイング」ボタンを表示するために再描画
        st.rerun()