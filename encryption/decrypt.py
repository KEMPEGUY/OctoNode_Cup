# encryption/decrypt.py

import sys
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet


def decrypt_file_content(encrypted_file_path):
    private_key_pem = os.environ.get("SUBMISSION_PRIVATE_KEY")

    if not private_key_pem:
        raise ValueError("SUBMISSION_PRIVATE_KEY missing")

    private_key_pem = private_key_pem.replace("\\n", "\n").strip()

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None
    )

    with open(encrypted_file_path, "rb") as f:
        file_content = f.read()

    # First 4 bytes = encrypted session key length
    key_length = int.from_bytes(file_content[:4], "big")

    encrypted_session_key = file_content[4:4 + key_length]
    encrypted_data = file_content[4 + key_length:]

    session_key = private_key.decrypt(
        encrypted_session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    cipher_suite = Fernet(session_key)
    decrypted_data = cipher_suite.decrypt(encrypted_data)

    return decrypted_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python decrypt.py <filename>")

    decrypted = decrypt_file_content(sys.argv[1])
    output_name = sys.argv[1].replace(".enc", ".csv")

    with open(output_name, "wb") as f:
        f.write(decrypted)

    print("DECRYPTION_SUCCESS")
