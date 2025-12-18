import streamlit as st
import os
import time
from pathlib import Path
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct
from openai import OpenAIError, APIError, APIConnectionError, RateLimitError

def record_audio(audio_input_file_path):
    """
    音声入力を受け取って音声ファイルを作成
    5秒間沈黙すると自動的に録音が完了します
    """

    audio_bytes = audio_recorder(
        key=f"audio_{audio_input_file_path}"
    )

    if audio_bytes:
        # 音声データをファイルに保存
        with open(audio_input_file_path, 'wb') as audio_file:
            audio_file.write(audio_bytes)
    else:
        st.stop()

def transcribe_audio(audio_input_file_path):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
    """
    try:
        with open(audio_input_file_path, 'rb') as audio_input_file:
            transcript = st.session_state.openai_obj.audio.transcriptions.create(
                model="whisper-1",
                file=audio_input_file,
                language="en"
            )
        
        # 音声入力ファイルを削除
        os.remove(audio_input_file_path)
        
        return transcript
        
    except RateLimitError:
        st.error("⚠️ API利用制限に達しました。しばらく待ってから再度お試しください。")
        st.stop()
    except APIConnectionError:
        st.error("⚠️ ネットワーク接続エラーが発生しました。インターネット接続を確認してください。")
        st.stop()
    except APIError as e:
        st.error(f"⚠️ OpenAI APIエラーが発生しました: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ 音声認識中にエラーが発生しました: {str(e)}")
        # エラーが発生してもファイルを削除
        if os.path.exists(audio_input_file_path):
            os.remove(audio_input_file_path)
        st.stop()

def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """

    temp_audio_output_filename = f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
        temp_audio_output_file.write(llm_response_audio)
    
    audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
    audio_mp3.export(audio_output_file_path, format="wav")

    # 音声出力用に一時的に作ったmp3ファイルを削除
    os.remove(temp_audio_output_filename)

def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ（Streamlitのst.audioを使用）
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """
    try:
        # 音声ファイルの読み込み
        audio = AudioSegment.from_wav(audio_output_file_path)
        
        # 速度を変更
        if speed != 1.0:
            # frame_rateを変更することで速度を調整
            modified_audio = audio._spawn(
                audio.raw_data, 
                overrides={"frame_rate": int(audio.frame_rate * speed)}
            )
            # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
            modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
            modified_audio.export(audio_output_file_path, format="wav")
        
        # 音声ファイルを読み込み
        with open(audio_output_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # Streamlitのaudio機能で再生
        st.audio(audio_bytes, format='audio/wav')
        
        # HTML5オーディオタグを使用して自動再生を試みる
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"⚠️ 音声再生中にエラーが発生しました: {str(e)}")
    finally:
        # 音声再生後に少し待ってからファイルを削除
        time.sleep(0.5)
        if os.path.exists(audio_output_file_path):
            os.remove(audio_output_file_path)

def create_chain(system_template):
    """
    LLMによる回答生成用のChain作成
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt
    )

    return chain

def create_problem_prompt_with_context(base_prompt, topic, situation):
    """
    トピックとシチュエーションを考慮した問題文生成プロンプトを作成
    Args:
        base_prompt: 基本のプロンプトテンプレート
        topic: 選択されたトピック
        situation: 選択されたシチュエーション
    """
    additional_context = ""
    
    if topic and topic != "ランダム":
        topic_desc = ct.TOPIC_OPTIONS.get(topic, "")
        if topic_desc:
            additional_context += f"\n    Focus on: {topic_desc}"
    
    if situation and situation != "指定なし":
        situation_desc = ct.SITUATION_OPTIONS.get(situation, "")
        if situation_desc:
            additional_context += f"\n    Situation: {situation_desc}"
    
    # 基本プロンプトに追加コンテキストを組み込む
    if additional_context:
        modified_prompt = base_prompt.rstrip() + additional_context + "\n"
        return modified_prompt
    
    return base_prompt

def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    Args:
        chain: 問題文生成用のChain
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
        openai_obj: OpenAIのオブジェクト
    """
    try:
        # 問題文を生成するChainを実行し、問題文を取得
        problem = st.session_state.chain_create_problem.predict(input="")

        # LLMからの回答を音声データに変換
        llm_response_audio = st.session_state.openai_obj.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=problem
        )

        # 音声ファイルの作成
        audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
        save_to_wav(llm_response_audio.content, audio_output_file_path)

        # 音声ファイルの読み上げ
        play_wav(audio_output_file_path, st.session_state.speed)

        return problem, llm_response_audio
        
    except RateLimitError:
        st.error("⚠️ API利用制限に達しました。しばらく待ってから再度お試しください。")
        st.stop()
    except APIConnectionError:
        st.error("⚠️ ネットワーク接続エラーが発生しました。インターネット接続を確認してください。")
        st.stop()
    except APIError as e:
        st.error(f"⚠️ OpenAI APIエラーが発生しました: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ 問題文生成中にエラーが発生しました: {str(e)}")
        st.stop()

def create_evaluation():
    """
    ユーザー入力値の評価生成
    """
    try:
        llm_response_evaluation = st.session_state.chain_evaluation.predict(input="")
        return llm_response_evaluation
        
    except RateLimitError:
        st.error("⚠️ API利用制限に達しました。しばらく待ってから再度お試しください。")
        st.stop()
    except APIConnectionError:
        st.error("⚠️ ネットワーク接続エラーが発生しました。インターネット接続を確認してください。")
        st.stop()
    except APIError as e:
        st.error(f"⚠️ OpenAI APIエラーが発生しました: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ 評価生成中にエラーが発生しました: {str(e)}")
        st.stop()

def extract_score(evaluation_text):
    """
    評価テキストからスコアを抽出
    Args:
        evaluation_text: 評価結果のテキスト
    Returns:
        score: スコア（0-100）、抽出できない場合はNone
    """
    import re
    
    # スコアのパターンを検索（例: "スコア: 85/100点"）
    score_pattern = r'スコア[:：]\s*(\d+)/100点'
    match = re.search(score_pattern, evaluation_text)
    
    if match:
        return int(match.group(1))
    return None

def display_score_badge(score):
    """
    スコアに応じたバッジを表示
    Args:
        score: スコア（0-100）
    """
    if score is None:
        return
    
    if score >= 90:
        st.success(f"🎉 素晴らしい！ スコア: {score}/100点")
    elif score >= 70:
        st.info(f"👏 良くできました！ スコア: {score}/100点")
    elif score >= 50:
        st.warning(f"📝 もう少し！ スコア: {score}/100点")
    else:
        st.error(f"💪 頑張りましょう！ スコア: {score}/100点")