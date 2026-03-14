# 🚀 Transformer Study Project Report: From Scratch to minGPT & Beyond

本プロジェクトは、Transformerアーキテクチャを構成要素（部品）からボトムアップで実装し、最終的にGPTモデル（Decoder-only）として学習・生成を行うまでの過程を記録したものです。
また、現代のLLM（Llama 3など）への技術的架け橋となる差分理解を含みます。

---

## 1. 🏗️ Build Process: 部品からの組み立て

ブラックボックスを避けるため、最小単位から実装し、テスト駆動（TDD）で動作を保証しながら積み上げました。

1.  **`MiniAttention` (The Core)**
    *   **実装**: `softmax(QK^T / √d)V` の数式をそのままコード化。
    *   **学び**: Attentionとは「検索（QueryでKeyを探し、Valueを取り出す）」であること。
2.  **`MultiHeadAttention` (Parallelism)**
    *   **実装**: 入力を分割し、複数の `MiniAttention` を並列走らせる。
    *   **学び**: 複数の「視点（Head）」を持つことで、文法・意味・位置など異なる情報を同時に処理できる。
3.  **`EncoderBlock` (The Body)**
    *   **実装**: Attention + FFN + Residual Connection + LayerNorm。
    *   **学び**: 残差接続（Shortcut）があるからこそ、勾配が消えずに深層学習が可能になる。
4.  **`GPT` (The Brain)**
    *   **実装**: Decoder-onlyアーキテクチャ。`Causal Mask`（未来を見ないマスク）と `Learned Positional Embedding` を導入。
    *   **学び**: 言語モデルとは「次の単語を予測する確率分布」そのものである。

---

## 2. 🧠 System Dynamics: `gpt.py` vs `train.py`

モデルの定義と学習プロセスは、以下のように密接に連携して「知能」を獲得しました。

| Component | Role | Analogy |
| :--- | :--- | :--- |
| **`gpt.py`** | **構造とパラメータの定義**<br>`wte` (単語辞書) や `wpe` (位置辞書)、Attentionの重みを保持する。 | **脳 (Brain)**<br>ニューロンの配線とシナプス結合の強さ。初期状態は「白紙（ランダム）」。 |
| **`train.py`** | **経験と修正のループ**<br>データを流し込み、誤差(`loss`)を計算し、パラメータを更新(`step`)する。 | **教育 (Education)**<br>教科書（シェイクスピア）を読ませ、間違いを指摘し、脳を書き換える教師。 |

### 📈 学習による進化 (Evolution)
`train.py` が `optimizer.step()` を呼ぶたびに、`gpt.py` 内の辞書が書き換わり、生成されるテキストが劇的に変化しました。

*   **Step 0**: `hesgwHyw...` (完全なノイズ)
*   **Step 40**: `the`, `would` (英単語の出現)
*   **Step 99**: `LARTIUS:`, `all` (シェイクスピアの戯曲形式を習得)

---

## 3. 🔍 Deep Dive: Data & Model Connections

`train.py` と `gpt.py` の具体的な連携ポイントと、データの実体について詳細を記録します。

### 3.1 📚 Where are the Data & Embeddings?

#### 学習データ (Input Text)
*   **Code**: `with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()`
*   **場所**: `input.txt` (カレントディレクトリにあるファイル)
*   **中身**: シェイクスピアの戯曲（約1MBのテキストデータ）
*   **役割**: これを読み込んで、モデルに「次に来る文字」を予測させるための正解データとして使います。

#### トークン辞書 (Token Embedding)
*   **Code**: `self.wte = nn.Embedding(config.vocab_size, config.n_embd)` (in `gpt.py`)
*   **場所**: `model.wte` (メモリ上のPyTorchテンソル)
*   **中身**: `[65, 384]` の行列（65種類の文字 × 384次元のベクトル）
*   **初期状態**: ランダムな数値
*   **学習中**: `optimizer.step()` で少しずつ書き換えられていきます。

#### 位置辞書 (Position Embedding)
*   **Code**: `self.wpe = nn.Embedding(config.block_size, config.n_embd)` (in `gpt.py`)
*   **場所**: `model.wpe` (メモリ上のPyTorchテンソル)
*   **中身**: `[256, 384]` の行列（最大256文字分の位置 × 384次元のベクトル）
*   **初期状態**: ランダムな数値
*   **学習中**: これも `optimizer.step()` で書き換えられていきます。

### 3.2 🔗 The 4 Connection Points (train.py ↔ gpt.py)

#### 接点 1: 生まれる瞬間 (Instantiation)
```python
# src/train.py
model = GPT(config)
```
*   ここで `gpt.py` の `__init__` が呼ばれます。
*   この瞬間、メモリ上に `model.wte`（単語辞書）と `model.wpe`（位置辞書）が作られます（中身はまだランダム）。
*   これ以降、`train.py` の変数 `model` は、あの辞書たちを「所有」しています。

#### 接点 2: 使われる瞬間 (Forward Pass)
```python
# src/train.py
logits, loss = model(xb, yb)
```
*   ここで `gpt.py` の `forward` メソッドが呼ばれます。
*   `xb`（入力の単語ID）が渡され、内部で `self.wte(idx)` が実行され、辞書からベクトルが取り出されます。
*   計算結果として `loss`（予測の外れ具合）が返ってきます。
*   **重要**: この `loss` は、ただの数字ではなく、**「どの辞書のどの値を使ったらこの数字になったか」という計算の履歴（グラフ）** を全部覚えています。

#### 接点 3: 犯人探しの瞬間 (Backward Pass)
```python
# src/train.py
loss.backward()
```
*   ここでPyTorchの魔法（自動微分）が発動します。
*   `loss` から計算履歴を逆のぼり、**「辞書のこの値を、これくらい増やせば Loss が減る」** という修正量（勾配: Gradient）を計算します。
*   この瞬間、`model.wte.weight.grad` という隠しポケットに、「修正指示書」がこっそり書き込まれます。まだ辞書自体は書き換わりません。

#### 接点 4: 書き換わる瞬間 (Optimizer Step)
```python
# src/train.py
optimizer.step()
```
*   ここでついに辞書が更新されます。
*   `optimizer` は `model.parameters()`（つまり `wte` や `wpe`）を管理するように設定されています。
*   `optimizer` が「修正指示書（grad）」を見て、**「よし、指示通りに辞書の数値をちょっとだけズラすぞ！」** と実行します。
*   **この瞬間に、`gpt.py` の中にある `self.wte` の数値が書き換わります。**

---

## 4. 👁️ Visualization: Attention Map

`notebooks/visualize_attention.ipynb` にて、モデル内部の思考プロセスを可視化しました。

*   **Sparsity (局所性)**: 特定の単語（自分自身や直前の単語）に強く注目するHeadが存在。
*   **Diversity (多様性)**: Headごとに全く異なる注目パターンを持つ（役割分担の証拠）。
*   **Mixing (混合)**: 層が深くなるにつれて情報が混ざり合い、文脈（Context）が形成される様子を確認。

---

## 5. 🌉 Bridge to Modern LLMs: Llamaへの足がかり

本プロジェクトの成果物 `doc/diff-map.md` は、今回作った「基本形 (minGPT)」と「最新形 (Llama 3)」の差分をマップ化したものです。

学校の課題や実務でLlama等を扱う際は、以下の **"Diff"** に注目することで、コードの意図を即座に理解できます。

*   **正規化**: `LayerNorm` → **`RMSNorm`** (計算軽量化)
*   **活性化**: `GELU` → **`SwiGLU`** (表現力向上)
*   **位置埋め込み**: `Learned Embedding` → **`RoPE`** (回転による相対位置記述・長文対応)
*   **Attention**: `Standard` → **`FlashAttention`** (爆速化)

---

## 🏁 Conclusion

このプロジェクトを通じて、Transformerは「魔法の箱」から**「行列演算と確率の集合体」**へと解像度が上がりました。

*   **Input**: IDの列
*   **Process**: 辞書を引き、混ぜ合わせ（Attention）、変換する（FFN）。
*   **Output**: 次の単語の確率。

この基礎理解があれば、パラメータが10億になろうと、アーキテクチャが多少変わろうと、恐れることはありません。
**You are ready for the Llama.**
