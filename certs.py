"""Самоподписанный сертификат для HTTPS в локальной сети (нужен для доступа к микрофону)."""
import datetime, ipaddress, os, socket
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def lan_ips():
    if os.environ.get("LAN_IPS"):  # в Docker задаётся снаружи: свой IP контейнера не подходит
        return sorted(i.strip() for i in os.environ["LAN_IPS"].split(",") if i.strip())
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def _write_key(path, k):
    open(path, "wb").write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                           serialization.NoEncryption()))


def ensure_ca(directory):
    """Свой локальный центр сертификации: ставится на устройства один раз, после чего HTTPS без предупреждений."""
    crt, key = os.path.join(directory, "ca.pem"), os.path.join(directory, "ca-key.pem")
    if os.path.exists(crt) and os.path.exists(key):
        return (x509.load_pem_x509_certificate(open(crt, "rb").read()),
                serialization.load_pem_private_key(open(key, "rb").read(), None), crt)
    k = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "IMCT Translator Local CA")])
    now = datetime.datetime.now(datetime.timezone.utc)
    ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
          .not_valid_after(now + datetime.timedelta(days=3650))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                                       encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(k.public_key()), critical=False)
          .sign(k, hashes.SHA256()))
    open(crt, "wb").write(ca.public_bytes(serialization.Encoding.PEM))
    _write_key(key, k)
    return ca, k, crt


def ensure_cert(directory):
    """Возвращает (cert, key, путь_к_CA). Сертификат сервера подписан локальным CA и пересоздаётся при смене IP."""
    os.makedirs(directory, exist_ok=True)
    ca, ca_key, ca_path = ensure_ca(directory)
    crt, key = os.path.join(directory, "cert.pem"), os.path.join(directory, "key.pem")
    ips = lan_ips()
    host = os.environ.get("HOST_NAME") or socket.gethostname()
    want = ["localhost", host, "127.0.0.1"] + ips
    if os.path.exists(crt) and os.path.exists(key):
        cur = x509.load_pem_x509_certificate(open(crt, "rb").read())
        san = cur.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        have = {str(v.value) for v in san}
        soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
        if set(want) <= have and cur.not_valid_after_utc > soon and cur.issuer == ca.subject:
            return crt, key, ca_path
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names = [x509.DNSName("localhost"), x509.DNSName(host), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    names += [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(ca.subject).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=397))  # iOS/Safari: не более 398 дней
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    open(crt, "wb").write(cert.public_bytes(serialization.Encoding.PEM))
    _write_key(key, k)
    return crt, key, ca_path
