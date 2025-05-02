import streamlit as st
import pandas as pd
import time
from datetime import datetime
import requests
from io import BytesIO
import tiktoken  # ✅ Tambahkan ini
from openai import OpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletion
from tqdm import tqdm
import ast
import logging
import os
import backoff
from typing import Optional, Union, List, Dict, Any



# === KONFIGURASI LOGGING ===
from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("app_log.txt", maxBytes=5_000_000, backupCount=2),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

encoding = tiktoken.encoding_for_model("gpt-4o")

try:
    openai_model = st.secrets["OPENAI_MODEL"]
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError as e:
    st.error(f"❌ Secrets belum di-set dengan benar: {e}")
    st.stop()


def process(
        df_raw_data: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Mulai proses klasifikasi issue dan sub-issue")

    # --- Filter hanya baris yang memenuhi syarat ---
    if "Issue" not in df_raw_data.columns:
        df_raw_data["Issue"] = ""
    if "Noise Tag" not in df_raw_data.columns:
        df_raw_data["Noise Tag"] = ""

    condition = (df_raw_data["Issue"].astype(str).str.strip() == "") | (df_raw_data["Noise Tag"].astype(str).str.strip() != "2")
    df_to_process = df_raw_data[condition].copy()

    logger.info(f"Jumlah baris yang diproses: {len(df_to_process)} dari total {len(df_raw_data)}")

    if df_to_process.empty:
        logger.warning("Tidak ada baris yang memenuhi kriteria untuk diproses.")
        return df_raw_data, 0, 0, 0


    if not all(col in df_raw_data.columns for col in ["Campaigns","Title","Content"]):
        logger.error("Column 'Campaigns' or 'Title' or 'Content' not found in the raw data.")
        return df_raw_data, 0, 0, 0
    else:
        
        list_issue, list_sub_issue, dict_sub_issue = [], [], {}
        completion_tokens, prompt_tokens, total_tokens = 0, 0, 0
        
        batch_content = token_based_split(df_to_process)
        batch_issue = []
        
        #print(len(batch_content[8]))
        for index, batch in tqdm(enumerate(batch_content), desc="STEP 1/4 Generate Issue", total=len(batch_content)):
            
            content_collection = "Campaign\tTitle\tContent\n"
            for campaign, title, content in batch:
                    content_collection += f"{campaign}\t{title}\t{content}\n"

            if index == 0:
                request_issue_sub_issue = request_chatgpt(
                    prompt_system=create_prompt_first_system(),
                    prompt_user=create_prompt_first_user(content_collection)
                )
                
                try:
                    issue = parse_response(request_issue_sub_issue)["issue"]
                    batch_issue.extend(issue)
            
                    completion_tokens += get_usage(request_issue_sub_issue)['completion_tokens']
                    prompt_tokens += get_usage(request_issue_sub_issue)['prompt_tokens']
                    total_tokens += get_usage(request_issue_sub_issue)['total_tokens']
                except Exception as e:
                    batch_issue.extend(["unknown"])
                    print(e)
                    continue
            else:
                _batch_issue = list(set(batch_issue))
                own_issue = "\n".join(_batch_issue)

                request_issue_sub_issue = request_chatgpt(
                    prompt_system=create_prompt_first_system_v2(own_issue),
                    prompt_user=create_prompt_first_user(content_collection)
                )
                try:
                    issue = parse_response(request_issue_sub_issue)['issue']
                    batch_issue.extend(issue)
            
                    completion_tokens += get_usage(request_issue_sub_issue)['completion_tokens']
                    prompt_tokens += get_usage(request_issue_sub_issue)['prompt_tokens']
                    total_tokens += get_usage(request_issue_sub_issue)['total_tokens']
                except Exception as e:
                    batch_issue.extend(["unknown"])
                    print(e)
                    continue
        
        list_issue = []
        for index, row in tqdm(df_to_process.iterrows(), desc="STEP 2/4 Select Issue", total=df_to_process.shape[0]):
            _batch_issue = list(set(batch_issue))
            own_issue = "\n".join(_batch_issue)

            content_collection = "Campaign\tTitle\tContent\n"
            content_collection += f"{row['Campaigns']}\t{row['Title']}\t{row['Content']}\n"

            request_issue_sub_issue = request_chatgpt(
                prompt_system=create_prompt_second_system(own_issue),
                prompt_user=create_prompt_second_user(content_collection)
            )

            try:
                issue = parse_response(request_issue_sub_issue)['issue']
                list_issue.append(issue)
            
                completion_tokens += get_usage(request_issue_sub_issue)['completion_tokens']
                prompt_tokens += get_usage(request_issue_sub_issue)['prompt_tokens']
                total_tokens += get_usage(request_issue_sub_issue)['total_tokens']
            except Exception as e:
                list_issue.append("unknown")
                print(e)
                continue

            ## CREATE CHECKPOINT
            if index % 10 == 0:  # Simpan setiap 10 iterasi
                try:
                    os.makedirs("checkpoints", exist_ok=True)
                    checkpoint_path = f"checkpoints/checkpoint_issue_sub_issue_method_2.xlsx"
                    checkpoint = pd.DataFrame({
                        'Campaigns': df_raw_data['Campaigns'].iloc[:len(list_issue)],
                        'Title': df_raw_data['Title'].iloc[:len(list_issue)],
                        'Content': df_raw_data['Content'].iloc[:len(list_issue)],
                        'Issue': list_issue,
                    })
                    checkpoint.to_excel(checkpoint_path, index=False)
                    logger.info(f"Checkpoint disimpan di: {checkpoint_path} (index ke-{index})")
                except Exception as e:
                    logger.error(f"Gagal menyimpan checkpoint: {str(e)}")
                    continue
        
        df_to_process['Issue'] = list_issue
        
        list_issue = list(set(df_raw_data["Issue"].dropna().tolist()))
        dict_sub_issue = {}
        for index, issue in tqdm(enumerate(list_issue), desc="STEP 3/4 Generate Sub Issue", total=len(list_issue)):
            if is_valid_issue(issue):
                filter_data = df_to_process[df_to_process['Issue'] == issue]

                batch_content = token_based_split(filter_data)
                batch_sub_issue = []

                for index, batch in enumerate(batch_content):
                    content_collection = "Campaign\tTitle\tContent\n"
                    for campaign, title, content in batch:
                            content_collection += f"{campaign}\t{title}\t{content}\n"
                    
                    if index == 0:
                        request_issue_sub_issue = request_chatgpt(
                            prompt_system=create_prompt_third_system(issue),
                            prompt_user=create_prompt_first_user(content_collection)
                        )
                        try:
                            sub_issue = parse_response(request_issue_sub_issue)["sub_issue"]
                            batch_sub_issue.extend(sub_issue)
            
                            completion_tokens += get_usage(request_issue_sub_issue)['completion_tokens']
                            prompt_tokens += get_usage(request_issue_sub_issue)['prompt_tokens']
                            total_tokens += get_usage(request_issue_sub_issue)['total_tokens']
                        except Exception as e:
                            batch_sub_issue.extend(["unknown"])
                            print(e)
                            continue
                    else:
                        _batch_sub_issue = list(set(batch_sub_issue))
                        own_sub_issue = "\n".join(_batch_sub_issue)

                        request_issue_sub_issue = request_chatgpt(
                            prompt_system=create_prompt_third_system_v2(issue, own_sub_issue),
                            prompt_user=create_prompt_first_user(content_collection)
                        )
                        try:
                            sub_issue = parse_response(request_issue_sub_issue)["sub_issue"]
                            batch_sub_issue.extend(sub_issue)
            
                            completion_tokens += get_usage(request_issue_sub_issue)['completion_tokens']
                            prompt_tokens += get_usage(request_issue_sub_issue)['prompt_tokens']
                            total_tokens += get_usage(request_issue_sub_issue)['total_tokens']
                        except Exception as e:
                            batch_sub_issue.extend(["unknown"])
                            print(e)
                            continue

                    dict_sub_issue[issue] = batch_sub_issue
        
        list_sub_issue = []
        for index, row in tqdm(df_to_process.iterrows(), desc="STEP 4/4 Select Sub Issue", total=df_to_process.shape[0]):
            if is_valid_issue(row['Issue']):
                _batch_issue = list(set(dict_sub_issue.get(row['Issue'], [])))
                own_issue = "\n".join(_batch_issue)

                content_collection = "Campaign\tTitle\tContent\n"
                content_collection += f"{row['Campaigns']}\t{row['Title']}\t{row['Content']}\n"

                request_issue_sub_issue = request_chatgpt(
                    prompt_system=create_prompt_fourth_system(own_issue),
                    prompt_user=create_prompt_second_user(content_collection)
                )

                try:
                    list_sub_issue.append(parse_response(request_issue_sub_issue)['sub_issue'])
                except Exception as e:
                    logger.error(f"Gagal parsing response Sub Issue pada index {index}: {str(e)}")
                    list_sub_issue.append("unknown")
                    continue
                
                ## CREATE CHECKPOINT
                if index % 10 == 0:  # Simpan setiap 10 iterasi
                    try:
                        os.makedirs("checkpoints", exist_ok=True)
                        checkpoint_path = f"checkpoints/checkpoint_issue_sub_issue_method_2.xlsx"
                        checkpoint = pd.DataFrame({
                            'Campaigns': df_raw_data['Campaigns'].iloc[:len(list_issue)],
                            'Title': df_raw_data['Title'].iloc[:len(list_issue)],
                            'Content': df_raw_data['Content'].iloc[:len(list_issue)],
                            'Issue': list_issue,
                        })
                        checkpoint.to_excel(checkpoint_path, index=False)
                        logger.info(f"Checkpoint disimpan di: {checkpoint_path} (index ke-{index})")
                    except Exception as e:
                        logger.error(f"Gagal menyimpan checkpoint: {str(e)}")
                        continue

                completion_tokens += get_usage(request_issue_sub_issue)['completion_tokens']
                prompt_tokens += get_usage(request_issue_sub_issue)['prompt_tokens']
                total_tokens += get_usage(request_issue_sub_issue)['total_tokens']
            else:
                list_sub_issue.append("unknown")
        
        df_to_process['Sub Issue'] = list_sub_issue

        price_input, price_output, total = estimate_cost(prompt_tokens, completion_tokens)

        df_raw_data.update(df_to_process)

        #print(f"\nPrice Input: ${price_input}")
        #print(f"Price Output: ${price_output}")
        #print(f"Total: ${total}")
        #print("This is only an estimate, the cost may not be accurate")
        
        #sub_issue_df = pd.DataFrame(dict_sub_issue.items(), columns=['Issue', 'Sub Issues'])
        #sub_issue_df.to_excel(f"sub_issues.xlsx", index=False)
        #return df_raw_data
        return df_raw_data, price_input, price_output, total



@backoff.on_exception(backoff.expo, RateLimitError, max_tries=10)  # Tambahkan decorator ini
def request_chatgpt(
        prompt_system: str,
        prompt_user: str,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.3,
        attempt: int = 1  # Tambahkan parameter untuk menghitung percobaan
) -> Union[ChatCompletion, Dict[str, str]]:
    try:
        response = client.chat.completions.create(
            model=openai_model,
            response_format= {
                "type": "json_object"
            },
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            n=n,
            stop=stop,
            temperature=temperature,
        )
        return response
    except RateLimitError as e:  # Tangkap RateLimitError di sini
        #logger.error(f"Kesalahan Rate Limit: {str(e)}")
        # Jika sudah mencapai max_tries, lanjutkan eksekusi
        if attempt >= 10:
            logger.info("Melewati batas max_tries, melanjutkan eksekusi.")
            return {"error": "rate limit", "message": "Melewati batas max_tries"}
        else:
            logger.error(f"Kesalahan Rate Limit: {str(e)}")
            return request_chatgpt(prompt_system, prompt_user, n, stop, temperature, attempt + 1)  # Coba lagi
    except OpenAIError as e:
        logger.error(f"Kesalahan OpenAI: {str(e)}")
        return {"error": "openai", "message": str(e)}
    except ValueError as e:
        logger.error(f"Kesalahan nilai: {str(e)}")
        return {"error": "value", "message": str(e)}
    except ConnectionError as e:
        logger.error(f"Kesalahan koneksi: {str(e)}")
        return {"error": "connection", "message": "Gagal terhubung ke server OpenAI"}
    except Exception as e:
        logger.error(f"Kesalahan tidak terduga: {str(e)}")
        return {"error": "unknown", "message": "Terjadi kesalahan yang tidak terduga"}


def create_prompt_fourth_system(sub_issue: str) -> str:
    return f"""Anda akan diberikan konten artikel yang dapat berupa konten berita ataupun post sosial media.
Tugas anda adalah mengidentifikasi salah satu sub isu utama dari beberapa sub isu berikut:
{sub_issue}\n
Jika teridentifikasi lebih dari 1 sub isu utama, pilih salah satu yang paling relevan.
Tinjau perubahan dan pastikan semua entri konsisten dan akurat.\n
Format the output in json object with 'sub_issue' as the key.
Make your response as short as possible.
"""

def create_prompt_third_system_v2(theme: str, own_sub_issue: str) -> str:
    return f"""Anda akan diberikan konten artikel atau post sosial media.
Tugas anda adalah, jika konten tersebut belum teridentifikasi dari sub issue berikut:
{own_sub_issue}\n
Maka identifikasi sub issue baru yang berbeda minimal 1 sub isu baru dari konten tersebut berdasarkan tema issue: '{theme}'.
Pastikan setiap sub issue yang diidentifikasi memiliki 2-10 kata.
Sub isu juga harus spesifik dari apa yang dimaksud sampai menyebutkan apa atau siapa yang terlibat.\n
Format the output in json object with 'sub_issue' as the key and the value as an array of strings.
Keep your response as short as possible."""

def create_prompt_third_system(theme: str) -> str:
    return f"""Anda akan diberikan kumpulan konten artikel atau post sosial media.
Konten berita atau post sosial media terdiri dari nama 'Campaign', 'Title' dan 'Content'.
Tugas anda adalah mengidentifikasi minimal 10 sub isu dari kumpulan konten tersebut berdasarkan tema issue: '{theme}'.
Pastikan setiap sub issue yang diidentifikasi memiliki 2-10 kata.
Sub isu juga harus spesifik dari apa yang dimaksud sampai menyebutkan apa atau siapa yang terlibat.\n
Format the output in json object with 'sub_issue' as the key and the value as an array of strings.
Keep your response as short as possible."""

def create_prompt_second_user(content: str) -> str:
    return f"""Berikut adalah data konten:\n
{content}"""

def create_prompt_second_system(issue: str) -> str:
    return f"""Anda akan diberikan konten artikel yang dapat berupa konten berita ataupun post sosial media.
Konten berita atau post sosial media terdiri dari nama 'Campaign', 'Title' dan 'Content'.
Tugas anda adalah mengidentifikasi hanya salah satu issue utama dari beberapa issue berikut:
{issue}\n
Jika teridentifikasi lebih dari 1 issue utama, pilih salah satu yang paling relevan.
Tinjau perubahan dan pastikan semua entri konsisten dan akurat.\n
Format the output in json object with 'issue' as the key.
Make your response as short as possible.
"""

def create_prompt_first_user(content_collection: str) -> str:
    return f"""Berikut adalah data kumpulan konten artikel atau post sosial media:\n
{content_collection}"""

def create_prompt_first_system_v2(own_issue: str) -> str:
    return f"""Anda akan diberikan kumpulan konten artikel atau post sosial media.
Konten berita atau post sosial media terdiri dari nama 'Campaign', 'Title' dan 'Content'.
Tugas anda adalah, jika konten tersebut belum teridentifikasi dari issue berikut:
{own_issue}\n
Maka identifikasi issue baru yang berbeda minimal 1 issue baru dari kumpulan konten artikel atau post sosial media tersebut.
Pastikan setiap issue yang diidentifikasi memiliki 2-3 kata.
Fokus mengelompokkan topik dan tema yang menonjol dari kumpulan konten artikel atau post sosial media.\n
Format the output in json object with 'issue' as the key and the value as an array of strings.
Keep your response as short as possible."""

def create_prompt_first_system() -> str:
    return f"""Anda akan diberikan kumpulan konten artikel atau post sosial media.
Konten berita atau post sosial media terdiri dari nama 'Campaign', 'Title' dan 'Content'.
Tugas anda adalah mengidentifikasi minimal 10 issue dari kumpulan konten tersebut.
Pastikan setiap issue yang diidentifikasi memiliki 2-3 kata.
Fokus mengelompokkan topik dan tema yang menonjol dari kumpulan konten artikel atau post sosial media.\n
Format the output in json object with 'issue' as the key and the value as an array of strings.
Keep your response as short as possible."""

def token_based_split(filter_data, max_tokens=100000):
    #encoding = tiktoken.encoding_for_model(model)
    all_content = filter_data[["Campaigns", "Title", "Content"]].astype(str).values.tolist()

    batch_content = []
    current_batch = []
    current_tokens = 0

    for campaign, title, content in all_content:
        content_tokens = count_tokens(content)

        if current_tokens + content_tokens > max_tokens:
            batch_content.append(current_batch)  # Menyimpan batch yang ada
            current_batch = [(campaign, title, content)]  # Memulai batch baru dengan campaign dan title
            current_tokens = content_tokens
        else:
            current_batch.append((campaign, title, content))  # Menambahkan campaign dan title ke batch
            current_tokens += content_tokens

    if current_batch:
        batch_content.append(current_batch)  # Menyimpan batch terakhir

    # Mengubah batch_content menjadi DataFrame sebelum mengembalikannya
    return batch_content

def count_tokens(text: str) -> int:
    try:
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())  # fallback kasar jika encoding gagal

def dict_to_readable_string(own_issue_sub_issue: Dict[str, List[str]]) -> str:
    result = ""
    for issue, sub_issues in own_issue_sub_issue.items():
        result += f"Issue: {issue}\n"
        for i, sub_issue in enumerate(sub_issues, 1):
            result += f"Sub Issue: {sub_issue}\n"
        result += "\n"
    return result

def parse_response(response: ChatCompletion) -> Dict[str, Any]:
    try:
        return ast.literal_eval(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"❌ Gagal parsing response GPT: {e}")
        return {"issue": "unknown", "sub_issue": "unknown"}

def is_valid_issue(issue):
    return (
        issue is not None
        and pd.notna(issue)
        and str(issue).strip() != ""
        and str(issue).strip() != "unknown"
        and not isinstance(issue, (float, int))  # Memastikan bukan angka
    )

def get_usage(response: ChatCompletion) -> Dict[str, int]:
    return {
        "completion_tokens": response.usage.completion_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens
    }

def estimate_cost(
        prompt_tokens: int,
        completion_tokens: int,
        price_input=0.150,
        price_output=0.600):
    """
    Menghitung estimasi biaya token berdasarkan jumlah prompt (input) dan completion (output) tokens.
    
    :param prompt_tokens: Jumlah token untuk input (prompt)
    :param completion_tokens: Jumlah token untuk output (completion)
    :param price_input: Harga per 1 juta input tokens (default: $0.150)
    :param price_output: Harga per 1 juta output tokens (default: $0.600)
    :return: Total estimasi biaya dalam USD
    """
    # Menghitung biaya input (prompt)
    biaya_input = (prompt_tokens / 1_000_000) * price_input
    
    # Menghitung biaya output (completion)
    biaya_output = (completion_tokens / 1_000_000) * price_output
    
    # Menghitung total biaya
    total_biaya = biaya_input + biaya_output
    
    return biaya_input, biaya_output, total_biaya


# === MULAI STREAMLIT APP ===
st.title("Insight Automation Phase 2")


# --- Try something
try:
    load_success = True
except Exception as e:
    load_success = False

if load_success:
    uploaded_raw = st.file_uploader("Upload Raw Data", type=["xlsx"], key="raw")
    submit = st.button("Submit", key="submit_button")


    if submit:
        if uploaded_raw is None:
            st.error("❌ Anda harus memilih upload raw data sebelum submit.")
        else:
            st.success(f"✅ File Loaded Successfully!")
            start_time = time.time()

            try:
                df_raw = pd.read_excel(uploaded_raw, sheet_name=0)
            except Exception as e:
                st.error(f"❌ Gagal membaca file Excel: {e}")
                st.stop()
            if "Campaign" in df_raw.columns:
                df_raw = df_raw.rename(columns={"Campaign": "Campaigns"})
            df_processed = df_raw.copy()
            
            #result = process(df_processed)
            result, price_input, price_output, total_price = process(df_processed)

            # Save Output
            # Ambil nama file asli tanpa ekstensi
            original_filename = uploaded_raw.name.rsplit(".", 1)[0]

            # Buat nama file output
            tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")
            output_filename = f"{original_filename}_phase2_{tanggal_hari_ini}.xlsx"


            #Jika keep raw data dan tidak keep raw data
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                result.to_excel(writer, sheet_name="Process Data", index=False)

            end_time = time.time()
            minutes, seconds = divmod(end_time - start_time, 60)

            # === Hitung durasi proses
            duration_seconds = end_time - start_time
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            st.info(f"🕒 Proses ini berjalan selama {int(hours)} jam {int(minutes)} menit {int(seconds)} detik.")
            
            logger.info("Proses selesai. Data dikembalikan.")

            # 1. Tampilkan Summary Execution Report
            st.subheader("📊 Summary Execution Report")
            with st.expander("📊 Lihat Summary Execution Report"):
                st.markdown(f"""
                **💰 Estimasi Biaya Token OpenAI:**

                - Price Input: **${price_input:.4f}**
                - Price Output: **${price_output:.4f}**
                - Total Estimate: **${total_price:.4f}**

                _Catatan: Ini hanya estimasi, biaya aktual mungkin berbeda._
                """)

            # 2. Tombol Download Hasil Excel (tetap tampil, tidak menghilang)
            with open(output_filename, "rb") as file:
                st.download_button(
                    label="⬇️ Download Hasil Excel",
                    data=file.read(),
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )

            # 3. Tombol download log file (jika ada)
            #if os.path.exists("app_log.txt"):
            #    with open("app_log.txt", "rb") as log_file:
            #        st.download_button(
            #            label="⬇️ Download Log File",
            #            data=log_file.read(),
            #            file_name="app_log.txt",
            #            mime="text/plain",
            #            key="download_log"
            #        )

            # 4. Tombol reset untuk memulai ulang proses
            #if st.button("🔄 Mulai dari Awal"):
            #    st.experimental_rerun()


else:
    st.stop()
