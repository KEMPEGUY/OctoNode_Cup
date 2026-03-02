import sys
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

def encrypt_file(input_file_path):
    _dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(_dir, "public_key.pem")
    with open(key_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())

    session_key = Fernet.generate_key()
    cipher_suite = Fernet(session_key)

    with open(input_file_path, "rb") as f:
        file_data = f.read()
    encrypted_data = cipher_suite.encrypt(file_data)

    encrypted_session_key = public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    output_file = input_file_path + ".enc"
    with open(output_file, "wb") as f:
        f.write(encrypted_session_key) 
        f.write(encrypted_data)
    
    print(f"Success! Encrypted to '{output_file}' (Hybrid Mode).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python encrypt.py <filename>")
    else:
        encrypt_file(sys.argv[1])
# ../submissions/baseline_model.csv --- IGNORE ---
