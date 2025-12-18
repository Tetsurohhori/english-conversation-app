# 更新内容詳細ドキュメント

## 実装日: 2025年12月17日

このドキュメントでは、英会話アプリに追加した5つの主要機能について、学習用に詳しく解説します。

---

## 📚 更新概要

### コミット情報
- **コミットID**: b029b5f
- **コミットメッセージ**: Add 5 major features: level selection, scoring system, learning history, error handling, and topic/situation selection
- **変更ファイル数**: 3ファイル
- **追加行数**: 365行
- **削除行数**: 73行

---

## 1️⃣ 英語レベル選択機能の実装

### 変更内容

#### constants.py
**変更前:**
```python
SYSTEM_TEMPLATE_BASIC_CONVERSATION = """単一の文字列プロンプト"""
SYSTEM_TEMPLATE_CREATE_PROBLEM = """単一の文字列プロンプト"""
```

**変更後:**
```python
SYSTEM_TEMPLATE_BASIC_CONVERSATION = {
    "初級者": """初級者向けプロンプト""",
    "中級者": """中級者向けプロンプト""",
    "上級者": """上級者向けプロンプト"""
}
SYSTEM_TEMPLATE_CREATE_PROBLEM = {
    "初級者": """初級者向け問題生成プロンプト""",
    "中級者": """中級者向け問題生成プロンプト""",
    "上級者": """上級者向け問題生成プロンプト"""
}
```

### 学習ポイント
- **辞書（Dictionary）の活用**: 単一の文字列から辞書型に変更し、英語レベルをキーにして適切なプロンプトを取得できるようにしました
- **動的なプロンプト選択**: ユーザーが選択した英語レベルに応じて、適切なプロンプトを動的に選択します

#### main.py
**追加した処理:**
```python
# 英語レベルに応じたChainを動的に作成
if st.session_state.chain_basic_conversation is None or 
   not hasattr(st.session_state, 'current_level') or 
   st.session_state.current_level != st.session_state.englv:
    st.session_state.chain_basic_conversation = ft.create_chain(
        ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION[st.session_state.englv]
    )
    st.session_state.current_level = st.session_state.englv
```

### 学習ポイント
- **条件分岐**: レベルが変更された場合のみChainを再作成することで、パフォーマンスを向上
- **hasattr()関数**: オブジェクトに属性が存在するかチェック
- **辞書のキーアクセス**: `dict[key]`形式でプロンプトを取得

---

## 2️⃣ 評価機能の強化（スコアリング・数値化）

### 変更内容

#### constants.py - 評価プロンプトの改善
**追加した分析項目:**
1. 単語の正確性（正しい単語数 / 総単語数）
2. 単語の順序の正確性
3. 文法的な正確性
4. スペルミスの有無
5. 抜け落ちた単語
6. 余分に追加された単語

**新しい出力フォーマット:**
```
【総合スコア】
スコア: XX/100点
正答率: XX%（正解した単語数/総単語数）

【詳細評価】
✓ 正確に再現できた部分
✗ 間違えた部分

【スコア内訳】
- 単語の正確性: XX/50点
- 語順・文法: XX/30点
- スペル: XX/20点
```

### 学習ポイント
- **構造化された出力**: LLMに明確なフォーマットを指示することで、パース可能な出力を得る
- **詳細な評価基準**: 複数の観点から評価することで、より具体的なフィードバックを提供

#### functions.py - スコア抽出とバッジ表示
**新規追加関数1: extract_score()**
```python
def extract_score(evaluation_text):
    """評価テキストからスコアを抽出"""
    import re
    score_pattern = r'スコア[:：]\s*(\d+)/100点'
    match = re.search(score_pattern, evaluation_text)
    if match:
        return int(match.group(1))
    return None
```

### 学習ポイント
- **正規表現（regex）**: パターンマッチングでテキストからデータを抽出
- **reモジュール**: `re.search()`でパターン検索、`match.group(1)`で最初のキャプチャグループを取得
- **エラー処理**: マッチしない場合はNoneを返す

**新規追加関数2: display_score_badge()**
```python
def display_score_badge(score):
    """スコアに応じたバッジを表示"""
    if score >= 90:
        st.success(f"🎉 素晴らしい！ スコア: {score}/100点")
    elif score >= 70:
        st.info(f"👏 良くできました！ スコア: {score}/100点")
    elif score >= 50:
        st.warning(f"📝 もう少し！ スコア: {score}/100点")
    else:
        st.error(f"💪 頑張りましょう！ スコア: {score}/100点")
```

### 学習ポイント
- **条件分岐（if-elif-else）**: スコアの範囲に応じて異なるメッセージを表示
- **Streamlitのステータスメッセージ**: success, info, warning, errorで視覚的なフィードバック
- **f-string**: 変数を埋め込んだ文字列の生成

#### main.py - スコア履歴の記録
```python
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
```

### 学習ポイント
- **セッション状態管理**: `st.session_state`でデータを保持
- **リストの動的追加**: `append()`でスコア履歴を蓄積
- **辞書の使用**: スコアをモード・タイムスタンプとセットで保存

---

## 3️⃣ 学習履歴・進捗管理機能の追加

### 変更内容

#### main.py - サイドバーの統計表示
```python
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
        import pandas as pd
        df_scores = pd.DataFrame({
            '回数': range(1, len(scores) + 1),
            'スコア': scores
        })
        st.line_chart(df_scores.set_index('回数'))
```

### 学習ポイント

#### リスト内包表記
```python
scores = [item['score'] for item in st.session_state.score_history]
```
- **リスト内包表記**: 効率的にリストから特定の値を抽出
- 通常のforループより簡潔で読みやすい

#### 統計計算
- `len(scores)`: リストの長さ = 学習回数
- `sum(scores) / len(scores)`: 平均値の計算
- `max(scores)`: 最大値の取得
- `scores[-1]`: リストの最後の要素（最新スコア）

#### Pandasデータフレーム
```python
df_scores = pd.DataFrame({
    '回数': range(1, len(scores) + 1),
    'スコア': scores
})
```
- **DataFrame**: 表形式のデータ構造
- **range()**: 連番の生成（1から開始）
- **set_index()**: 指定した列をインデックスに設定

#### Streamlitのグラフ表示
```python
st.line_chart(df_scores.set_index('回数'))
```
- **折れ線グラフ**: スコアの推移を可視化
- DataFrameを直接グラフ化

#### モード別統計
```python
mode_stats = {}
for item in st.session_state.score_history:
    mode = item['mode']
    if mode not in mode_stats:
        mode_stats[mode] = []
    mode_stats[mode].append(item['score'])
```

### 学習ポイント
- **辞書の動的構築**: キーが存在しない場合に初期化
- **グループ化**: モードごとにスコアを集約
- **forループ**: リストの要素を1つずつ処理

---

## 4️⃣ エラーハンドリングの実装

### 変更内容

#### functions.py - OpenAIエラーのインポート
```python
from openai import OpenAIError, APIError, APIConnectionError, RateLimitError
```

#### エラーハンドリングの実装例（transcribe_audio関数）
```python
def transcribe_audio(audio_input_file_path):
    try:
        with open(audio_input_file_path, 'rb') as audio_input_file:
            transcript = st.session_state.openai_obj.audio.transcriptions.create(
                model="whisper-1",
                file=audio_input_file,
                language="en"
            )
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
        if os.path.exists(audio_input_file_path):
            os.remove(audio_input_file_path)
        st.stop()
```

### 学習ポイント

#### try-except構文
```python
try:
    # 実行したい処理
except 特定のエラー:
    # エラー時の処理
```
- **例外処理**: エラーが発生してもプログラムが停止しないようにする
- **複数のexcept**: エラーの種類ごとに異なる処理を実装

#### エラーの種類と対応
1. **RateLimitError**: API使用量の制限超過
2. **APIConnectionError**: ネットワーク接続の問題
3. **APIError**: API固有のエラー（as eで詳細を取得）
4. **Exception**: その他すべてのエラー（最も広範囲）

#### st.stop()の使用
- Streamlitの実行を停止
- エラー発生時にそれ以降の処理を実行しない

#### リソースのクリーンアップ
```python
if os.path.exists(audio_input_file_path):
    os.remove(audio_input_file_path)
```
- エラーが発生してもファイルを削除
- リソースリークを防ぐ

#### main.py - 日常英会話モードのエラーハンドリング
```python
try:
    llm_response = st.session_state.chain_basic_conversation.predict(input=audio_input_text)
    llm_response_audio = st.session_state.openai_obj.audio.speech.create(...)
    ft.save_to_wav(llm_response_audio.content, audio_output_file_path)
except Exception as e:
    st.error(f"⚠️ 回答生成中にエラーが発生しました: {str(e)}")
    st.stop()
```

### 学習ポイント
- **包括的なエラー処理**: 複数の処理を1つのtry-exceptでまとめて処理
- **ユーザーフレンドリーなメッセージ**: 技術的な詳細を隠して分かりやすく表示

---

## 5️⃣ 問題文のバリエーション追加（トピック・シチュエーション選択）

### 変更内容

#### constants.py - 選択肢の定義
```python
TOPIC_OPTIONS = {
    "ランダム": "",
    "日常会話": "daily conversation topics such as hobbies, weather, family, food",
    "ビジネス": "business topics such as meetings, emails, presentations, negotiations",
    "旅行": "travel topics such as hotels, airports, restaurants, sightseeing",
    # ... 他のトピック
}

SITUATION_OPTIONS = {
    "指定なし": "",
    "自己紹介": "introducing yourself to someone new",
    "レストラン": "ordering food at a restaurant",
    # ... 他のシチュエーション
}
```

### 学習ポイント
- **辞書のキーと値**: 日本語表示（キー）と英語の説明（値）を対応付け
- **空文字列**: デフォルト値として空文字列を使用

#### functions.py - プロンプト生成関数
```python
def create_problem_prompt_with_context(base_prompt, topic, situation):
    """トピックとシチュエーションを考慮した問題文生成プロンプトを作成"""
    additional_context = ""
    
    if topic and topic != "ランダム":
        topic_desc = ct.TOPIC_OPTIONS.get(topic, "")
        if topic_desc:
            additional_context += f"\n    Focus on: {topic_desc}"
    
    if situation and situation != "指定なし":
        situation_desc = ct.SITUATION_OPTIONS.get(situation, "")
        if situation_desc:
            additional_context += f"\n    Situation: {situation_desc}"
    
    if additional_context:
        modified_prompt = base_prompt.rstrip() + additional_context + "\n"
        return modified_prompt
    
    return base_prompt
```

### 学習ポイント

#### 条件判定
```python
if topic and topic != "ランダム":
```
- **論理演算子**: topicが存在し、かつ「ランダム」でない場合

#### 辞書のget()メソッド
```python
topic_desc = ct.TOPIC_OPTIONS.get(topic, "")
```
- **安全な値の取得**: キーが存在しない場合はデフォルト値（空文字列）を返す
- `dict[key]`と異なり、KeyErrorが発生しない

#### 文字列の連結
```python
additional_context += f"\n    Focus on: {topic_desc}"
```
- **複合代入演算子（+=）**: 既存の文字列に追加
- **改行文字（\n）**: テキストの整形

#### 文字列のrstrip()メソッド
```python
modified_prompt = base_prompt.rstrip() + additional_context + "\n"
```
- **rstrip()**: 文字列の末尾の空白を削除
- プロンプトの整形に使用

#### main.py - UIの追加
```python
if st.session_state.mode in [ct.MODE_2, ct.MODE_3]:
    col5, col6 = st.columns(2)
    with col5:
        st.session_state.topic = st.selectbox(
            label="トピック",
            options=list(ct.TOPIC_OPTIONS.keys()),
            label_visibility="collapsed",
            key="topic_select"
        )
    with col6:
        st.session_state.situation = st.selectbox(
            label="シチュエーション",
            options=list(ct.SITUATION_OPTIONS.keys()),
            label_visibility="collapsed",
            key="situation_select"
        )
```

### 学習ポイント

#### リスト包含演算子（in）
```python
if st.session_state.mode in [ct.MODE_2, ct.MODE_3]:
```
- **in演算子**: リスト内に要素が含まれるかチェック
- 複数の条件をコンパクトに記述

#### 辞書のkeys()メソッド
```python
options=list(ct.TOPIC_OPTIONS.keys())
```
- **keys()**: 辞書のすべてのキーを取得
- **list()**: キーをリストに変換（Streamlitのselect boxに必要）

#### Streamlitのレイアウト
```python
col5, col6 = st.columns(2)
```
- **カラムレイアウト**: 画面を2列に分割
- `with col5:`ブロックで各カラムに要素を配置

#### st.selectboxの引数
- **label**: ラベルテキスト
- **options**: 選択肢のリスト
- **label_visibility**: ラベルの表示/非表示
- **key**: セッション状態のキー（一意である必要がある）

#### プロンプトへの適用
```python
base_prompt = ct.SYSTEM_TEMPLATE_CREATE_PROBLEM[st.session_state.englv]
topic = st.session_state.get('topic', 'ランダム')
situation = st.session_state.get('situation', '指定なし')
modified_prompt = ft.create_problem_prompt_with_context(base_prompt, topic, situation)
st.session_state.chain_create_problem = ft.create_chain(modified_prompt)
```

### 学習ポイント
- **段階的な処理**: 基本プロンプト → トピック/シチュエーション追加 → Chain作成
- **get()メソッドの活用**: デフォルト値を指定して安全に値を取得
- **関数の組み合わせ**: 複数の関数を連携して目的を達成

---

## 🎯 全体のアーキテクチャ

### ファイルの役割分担

1. **constants.py**: 定数・プロンプトの定義
   - 変更が容易
   - 一元管理

2. **functions.py**: ビジネスロジック
   - 再利用可能な関数
   - エラーハンドリング

3. **main.py**: UI・フロー制御
   - ユーザーインタラクション
   - 状態管理

### データフロー

```
ユーザー入力（UI）
    ↓
セッション状態に保存
    ↓
適切なプロンプト選択
    ↓
LLM API呼び出し
    ↓
エラーハンドリング
    ↓
結果の処理・表示
    ↓
履歴に記録
```

---

## 💡 学んだPythonの概念

### データ構造
- **リスト**: 順序付きコレクション
- **辞書**: キーと値のペア
- **タプル**: 不変の順序付きコレクション

### 制御構文
- **if-elif-else**: 条件分岐
- **for**: 繰り返し処理
- **try-except**: 例外処理

### 文字列操作
- **f-string**: フォーマット済み文字列
- **rstrip()**: 末尾の空白削除
- **正規表現**: パターンマッチング

### 関数
- **def**: 関数定義
- **引数とデフォルト値**: 柔軟な関数設計
- **戻り値**: return文

### モジュール
- **import**: 外部ライブラリの読み込み
- **from ... import**: 特定の機能のみインポート

### Streamlit固有
- **st.session_state**: セッション状態管理
- **st.sidebar**: サイドバー
- **st.columns**: レイアウト
- **st.metric**: メトリクス表示
- **st.line_chart**: グラフ表示

---

## 🔧 デバッグとテストのヒント

### エラー確認方法
```bash
# ログの確認
streamlit run main.py

# Gitの差分確認
git diff ffff204 b029b5f

# 特定のファイルの差分
git diff ffff204 b029b5f -- functions.py
```

### 動作確認のポイント
1. 英語レベルを変更して問題の難易度が変わるか
2. スコアが正しく表示されるか
3. サイドバーの統計が更新されるか
4. エラー時に適切なメッセージが表示されるか
5. トピック/シチュエーション選択が問題に反映されるか

---

## 📚 参考資料

### Python公式ドキュメント
- 正規表現: https://docs.python.org/ja/3/library/re.html
- 例外処理: https://docs.python.org/ja/3/tutorial/errors.html
- データ構造: https://docs.python.org/ja/3/tutorial/datastructures.html

### Streamlit公式ドキュメント
- API リファレンス: https://docs.streamlit.io/library/api-reference
- セッション状態: https://docs.streamlit.io/library/api-reference/session-state

### LangChain公式ドキュメント
- ConversationChain: https://python.langchain.com/docs/modules/memory/

---

## ✅ チェックリスト

学習の確認用に以下を試してみてください：

- [ ] constants.pyの辞書型プロンプトの構造を理解した
- [ ] functions.pyの新しい関数の動作を理解した
- [ ] main.pyのUIとロジックの連携を理解した
- [ ] try-except構文の使い方を理解した
- [ ] 正規表現でのパターンマッチングを理解した
- [ ] Streamlitのセッション状態管理を理解した
- [ ] Pandasデータフレームの基本を理解した
- [ ] 実際にコードを動かして各機能を確認した

---

## 🚀 次のステップ

さらに学習を深めるための提案：

1. **コードの読解**: 各関数を1つずつ読んで、処理の流れを追う
2. **小さな改造**: 例えばスコアの閾値を変更してみる
3. **新機能の追加**: 学んだ知識を応用して新しい機能を実装
4. **エラー発生実験**: 意図的にエラーを発生させて、ハンドリングを確認
5. **ドキュメント作成**: 自分なりにコードの説明を書いてみる

---

このドキュメントは学習用に作成されました。
質問や不明点があれば、いつでもお尋ねください！



