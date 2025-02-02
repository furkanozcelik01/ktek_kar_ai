
# Konsantrasyon Takip Uygulaması

Bu uygulama, kullanıcının konsantrasyon seviyesini kamera üzerinden yüz ve göz hareketlerini analiz ederek takip eden bir Python uygulamasıdır.

## Özellikler

- Gerçek zamanlı göz takibi
- Baş pozisyonu analizi
- Konsantrasyon seviyesi ölçümü
- Görsel geri bildirim sistemi
- Tam ekran desteği

## Gereksinimler
pip install -r requirements.txt

## Kurulum

1. Projeyi klonlayın:

git clone https://github.com/kullaniciadi/konsantrasyon-takip.git
cd konsantrasyon-takip

2. Gerekli paketleri yükleyin:

pip3 install -r requirements.txt --break-system-packages --no-warn-script-location

3. Qt kaynak dosyalarını derleyin:

pyside6-rcc res.qrc -o res_rc.py

## Kontroller

- **ESC**: Uygulamadan çıkış
- **Kapat Butonu**: Uygulamayı sonlandır

## Nasıl Çalışır?

python main.py

1. Uygulama kamera görüntüsünü alır
2. MediaPipe kullanarak yüz ve göz tespiti yapar
3. Göz kırpma oranı ve baş pozisyonunu analiz eder
4. Konsantrasyon seviyesini hesaplar ve görsel geri bildirim sağlar

##  NOTE
Python 3.12.7

sudo apt install python-is-python3 (python3 yerine python kullanma)* 
