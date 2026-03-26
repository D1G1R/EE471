def please_conform_clean(caps: list[str]) -> None:
    """
    Dizideki tüm elemanları aynı yöne çevirmek için gereken minimum
    komutları hesaplar ve yazdırır. (Clean Code yaklaşımı ile optimize edilmiştir)
    """
    if not caps:
        return
    
    start = 0
    # Her zaman dizinin ilk elemanından FARKLI olan yönü hedef alıyoruz.
    target_cap = 'B' if caps[0] == 'F' else 'F'
    
    # Döngünün son bloğu da yakalayabilmesi için geçici bir sonlandırıcı ekliyoruz
    caps.append('End') 
    
    for i in range(1, len(caps)):
        # Şapka yönü değiştiğinde:
        if caps[i] != caps[i - 1]:
            # Eğer biten blok bizim çevirmek istediğimiz hedef blok ise yazdır:
            if caps[i - 1] == target_cap:
                if start == i - 1:
                    print(f"Person in position {start} flip your cap!")
                else:
                    print(f"People in positions {start} through {i - 1} flip your caps!")
            
            # Yeni bloğun başlangıç indeksini güncelle
            start = i
            
    # Listeyi orijinal haline geri döndür
    caps.pop()

if __name__ == "__main__":
    caps_list = ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'B', 'F']
    print("Testing first caps list:")
    please_conform_clean(caps_list)