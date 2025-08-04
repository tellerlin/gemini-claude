import os
import sys
import time
import re
from typing import List, Tuple, Set

# --- Dependency Check ---
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from dotenv import load_dotenv
except ImportError:
    print("错误：必要的库未安装。")
    print("请运行以下命令来安装此脚本所需的依赖：")
    print("pip install python-dotenv google-generativeai")
    sys.exit(1)

# ANSI color codes for colorful output in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Gemini API Key format regex, matches keys like "AIzaSy..."
GEMINI_KEY_PATTERN = re.compile(r"^AIzaSy[A-Za-z0-9_-]{33}$")

def check_gemini_api_key(api_key: str) -> Tuple[str, str]:
    """
    Checks if a single Google Gemini API key can return a value (has quota).

    Args:
        api_key: The API key to check.

    Returns:
        A tuple containing the key's status ('valid', 'temporarily invalid',
        'permanently invalid') and a descriptive message.
    """
    try:
        genai.configure(api_key=api_key)
        
        # --- MODIFIED PART ---
        # Instead of listing models, we now perform a minimal content generation
        # to accurately check for quota and usage limits.
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        model.generate_content(
            'test', 
            generation_config=genai.types.GenerationConfig(max_output_tokens=1)
        )
        # --- END MODIFIED PART ---

        return 'valid', 'Key is valid and has sufficient quota.'
    except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated) as e:
        return 'permanently invalid', f'Authentication failed. Key is invalid or disabled. (Reason: {getattr(e, "message", str(e))})'
    except google_exceptions.ResourceExhausted as e:
        # This is the key change: a 429 error now correctly identifies a key that is out of quota.
        return 'temporarily invalid', f'Quota exceeded or rate limit hit. (Reason: {getattr(e, "message", str(e))})'
    except google_exceptions.DeadlineExceeded:
        return 'temporarily invalid', 'Request timed out. Could be a temporary issue with the API or network.'
    except google_exceptions.ServiceUnavailable:
        return 'temporarily invalid', 'The Google AI service is currently unavailable. Please try again later.'
    except google_exceptions.GoogleAPICallError as e:
        return 'permanently invalid', f'An unexpected API error occurred (Code: {getattr(e, "code", "N/A")}). Key might be misconfigured. (Reason: {getattr(e, "message", str(e))})'
    except Exception as e:
        # Catch any other unexpected exceptions (like network issues)
        return 'temporarily invalid', f'An unexpected network or client-side error occurred: {e}'

def update_env_file(keys_to_keep: List[str], original_env_path: str = '.env'):
    """
    Creates a new .env file with the updated list of keys, preserving all other content.
    """
    new_env_path = f"{original_env_path}.updated"
    updated_keys_str = ",".join(keys_to_keep)

    try:
        with open(original_env_path, 'r') as f_in, open(new_env_path, 'w') as f_out:
            for line in f_in:
                if line.strip().startswith('GEMINI_API_KEYS='):
                    f_out.write(f'GEMINI_API_KEYS={updated_keys_str}\n')
                else:
                    f_out.write(line)
    except FileNotFoundError:
        print(f"\n{bcolors.FAIL}Error: The original '{original_env_path}' file was not found.{bcolors.ENDC}")
        print("Creating a new '.env.updated' file with only the keys to keep.")
        with open(new_env_path, 'w') as f_out:
            f_out.write('# Please add back any other necessary .env variables\n')
            f_out.write(f'GEMINI_API_KEYS={updated_keys_str}\n')

    print(f"\n{bcolors.OKGREEN}✅ Success! A new file `.env.updated` has been created.{bcolors.ENDC}")
    print(f"It contains your original settings with the updated API keys.")
    print(f"Please review the new file. If it's correct, you can delete your old `.env` file and rename `.env.updated` to `.env`.")

def main():
    """
    Main function to run the API key checker tool.
    """
    load_dotenv()

    api_keys_str = os.getenv("GEMINI_API_KEYS", "")
    initial_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]

    if not initial_keys or initial_keys == ['your-google-ai-api-key-1', 'your-google-ai-api-key-2']:
        print(f"{bcolors.FAIL}Error: No valid API keys found in your .env file.{bcolors.ENDC}")
        print("Please replace the placeholder values in `GEMINI_API_KEYS` with your actual keys.")
        sys.exit(1)

    print(f"{bcolors.HEADER}--- Step 1: Pre-processing Keys ---{bcolors.ENDC}")
    print(f"Found {len(initial_keys)} key(s) in .env file.")

    valid_format_keys, invalid_format_keys = [], []
    for key in initial_keys:
        if GEMINI_KEY_PATTERN.match(key):
            valid_format_keys.append(key)
        else:
            invalid_format_keys.append(key)

    if invalid_format_keys:
        print(f"{bcolors.WARNING}Warning: Found {len(invalid_format_keys)} key(s) with an invalid format. They will be ignored:{bcolors.ENDC}")
        for key in invalid_format_keys: print(f"  - {key}")

    unique_keys = list(dict.fromkeys(valid_format_keys))
    if len(valid_format_keys) > len(unique_keys):
        print(f"{bcolors.OKCYAN}Removed {len(valid_format_keys) - len(unique_keys)} duplicate key(s).{bcolors.ENDC}")

    if not unique_keys:
        print(f"{bcolors.FAIL}\nAfter pre-processing, no valid formatted keys remain to be checked. Exiting.{bcolors.ENDC}")
        sys.exit(1)

    print(f"Proceeding to check {len(unique_keys)} unique, valid-format key(s).")
    print(f"\n{bcolors.HEADER}--- Step 2: Validating Keys via API (1 sec delay per key) ---{bcolors.ENDC}")

    categorized_keys = {'valid': [], 'temporarily invalid': [], 'permanently invalid': invalid_format_keys}

    for idx, key in enumerate(unique_keys):
        key_display = f"{key[:7]}...{key[-4:]}"
        print(f"[{idx+1}/{len(unique_keys)}] Checking key {bcolors.BOLD}{key_display}{bcolors.ENDC}...", end="", flush=True)

        status, message = check_gemini_api_key(key)
        categorized_keys[status].append(key)

        color_map = {'valid': bcolors.OKGREEN, 'temporarily invalid': bcolors.WARNING, 'permanently invalid': bcolors.FAIL}
        print(f"\r[{idx+1}/{len(unique_keys)}] Key {bcolors.BOLD}{key_display}{bcolors.ENDC}: {color_map[status]}{status.upper()}{bcolors.ENDC} - {message.splitlines()[0]}")

        if idx < len(unique_keys) - 1:
            time.sleep(1)

    print("\n" + "="*22 + " SUMMARY " + "="*22)
    print(f"{bcolors.OKGREEN}Valid keys (can return value): {len(categorized_keys['valid'])}{bcolors.ENDC}")
    print(f"{bcolors.WARNING}Temporarily invalid keys (e.g., out of quota): {len(categorized_keys['temporarily invalid'])}{bcolors.ENDC}")
    print(f"{bcolors.FAIL}Permanently invalid keys: {len(categorized_keys['permanently invalid'])} (includes format-invalid keys){bcolors.ENDC}")
    print("="*53 + "\n")

    try:
        while True:
            print(f"{bcolors.HEADER}--- Step 3: Update .env File ---{bcolors.ENDC}")
            print("1: Keep ONLY the valid keys.")
            print("2: Keep both valid and temporarily invalid keys.")
            print("3: Exit without making any changes.")

            choice = input("Enter your choice (1, 2, or 3): ").strip()
            keys_to_keep = []
            action_text = ""

            if choice == '1':
                keys_to_keep, action_text = categorized_keys['valid'], "Action: Keep only valid keys."
            elif choice == '2':
                keys_to_keep, action_text = categorized_keys['valid'] + categorized_keys['temporarily invalid'], "Action: Keep valid and temporarily invalid keys."
            elif choice == '3':
                print("\nExiting without any changes.")
                sys.exit(0)
            else:
                print(f"{bcolors.FAIL}Invalid choice. Please enter 1, 2, or 3.{bcolors.ENDC}\n")
                continue

            if not keys_to_keep:
                print(f"{bcolors.FAIL}\nError: Your choice would result in zero keys being saved. This is not allowed.{bcolors.ENDC}")
                print("No changes will be made. Please add at least one valid key and re-run.\n")
                continue

            print(action_text)
            update_env_file(keys_to_keep)
            break
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
