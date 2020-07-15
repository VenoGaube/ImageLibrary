# Pogon skripte  
  
1.  git clone https://github.com/VenoGaube/ImageLibrary.git  
2.  cd ImageLibrary  
3.  pip install -r requirements.txt  
4.  python UserInterface.py  
  
# Izvajanje UserInterface.py  
  
1. Program pregleda vse možne folderje na računalniku (ampak samo na enem drive-u za zdaj) in shrani slike v array.  
2. Ko vse slike najde, jih vse shrani v mapo ImageLibrary/facenet/src/data/test_raw/gallery.  
3. Nato bo v while True loop-u program prikazoval naključne slike iz selekcije slik in jih glede na vnos v polju kategoriziral v mapo pod 
ImageLibrary/facenet/src/data/train_raw/$IME_OSEBE. Kako se ime osebe napiše ni pomembno, saj se vedno pretvori v .upper(), tako da je shranjeno 
vedno v isti folder.  
4. Ko je v vsakem /$IME_OSEBE folderju vsaj 20 slik, se bo vnosno okno in okno s sliko zaprlo in prične se avtomatsko testiranje, ki traja od 5-20 minut, 
odvisno od količine slik znotraj ../test_raw/gallery folderja.  
5. Prikazani so progression bar-i, da se grafično vidi kako daleč je program s kakšnim ukazom.  
6. Ob koncu izvajanja vseh 4. avtomatiziranih ukazov, se bo v datoteki ../ImageLibrary ustvarila datoteka /results, kamor bodo shranjene originalne slike oseb 
znotraj njim določenim folderjev. Torej oseba "Robert" bo imela znotraj ../ImageLibrary/results/ROBERT shranjene vse njegove slike. Slika se znajde znotraj folderja
/results/$IME_OSEBE, le če je prepričanost programa večja ali enaka 0.900.  

