
import pytesseract                  #pour faire de la reconnaissance de caracteres
import cv2                          #pour appliquer des methodes de TIM
import numpy as np                  #pour appliquer des methodes de TIM
import matplotlib.pyplot as plt     #pour afficher une image
import math                         #pour utiliser floor et ceil
import pylab                        #pour afficher des courbes
import os                           #pour avoir les paths
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print()


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def ocr_core(image):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(image)  
    return text

def get_sub_image(image,coord):
    """
    this function return the part of an image giving the coord of the selection
    the crop region must be given as a 4-tuple - (left, upper, right, lower).
    """
    x1 = coord[0]
    y1 = coord[1]
    x2 = coord[2]
    y2 = coord[3]
    cropped_im = image[y1:y2, x1:x2]
    return cropped_im

def delimited_ocr(image,coord):
    imageReduite = get_sub_image(image, coord)
    #affiche_image(imageReduite)   # decommenter pour voir les zones qui sont selectionnés
    return ocr_core(imageReduite)

def affiche_image(image):
    plt.imshow(image)
    plt.show()


def affiche_zone(image,coord):
    zone = get_sub_image(image,coord)
    affiche(zone)

def size_carre(carre):
    """
    renvoie la longueur de la diagonale d'un carré avec
    les coordonées de 2 points opposés du carré
    """
    point1, point2 = carre
    x1, y1 = point1
    x2, y2 = point2
    diagonale = ((x1-x2)**2 + (y1-y2)**2)**0.5
    diagonale = math.floor(diagonale)
    return diagonale

def centre_carre(carre):
    """
    renvoie la position du centre du carré
    avec les coordonées de 2 points opposés du carré
    """
    point1, point2 = carre
    x1, y1 = point1
    x2, y2 = point2
    x3 = (x1+x2)/2
    y3 = (y1+y2)/2
    x3 = math.floor(x3)
    y3 = math.floor(y3)
    point3 = (x3,y3)
    return point3

def plus_petit(listeCarre):
    minimum = 50000
    for carre in listeCarre:
        taille = size_carre(carre)
        if(taille < minimum):
            minimum = taille
            pluspetitcarre = carre
    return pluspetitcarre


def detect_carre(image):
    """
    renvoi la position des carrés sur la page
    """

    contours, ret = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #recupere tous les carres de la page
    listeCarres =[]
    for cnt in contours:
        cv2.drawContours(gray, [cnt], 0 , 0)
        approx = cv2.approxPolyDP(cnt , 0.05 * cv2.arcLength(cnt, True), True)  #nombre de coté du polygone

        if len(approx) == 4:
            point1 = (approx.ravel()[0], approx.ravel()[1])
            point2 = (approx.ravel()[4], approx.ravel()[5])
            carre = (point1, point2)
            listeCarres.append(carre)

    return listeCarres

  
def select_bon_carre(listeCarres):

    """
    renvoie la position des 4 plus petits carré de la liste des carres,
    ce sont correspondant au numero de page
    il faut que la liste ait au moins 4 carrés 

    """

    #tri pour garder seuleemnt les 4 petits carré correspond aux numeros de page
    listePlusPetitCarres = []
    for i in range(0,4):
        listePlusPetitCarres.append(plus_petit(listeCarres))
        listeCarres.remove(plus_petit(listeCarres))

    #garde uniquement les coordonées
    listeCoordCarre = []
    for carre in listePlusPetitCarres:
        listeCoordCarre.append(centre_carre(carre))

    return listeCoordCarre


def detect_centre_motif(listeCoordCarre):
    """
    determine la position du motif en sachant les positions des 4 petits carres
    """
    carre1 = listeCoordCarre[0]
    carre2 = listeCoordCarre[1]
    carre3 = listeCoordCarre[2]
    carre4 = listeCoordCarre[3]

    grandCarre1 = (carre1,carre2)
    grandCarre2 = (carre3,carre4)
    grandCarre1 = centre_carre(grandCarre1)
    grandCarre2 = centre_carre(grandCarre2)

    carreFinal = (grandCarre1, grandCarre2)
    return centre_carre(carreFinal)

def detect_taille_motif(listeCoordCarre):
    """
    calcule la taille de la forme regroupant les 4 carrées
    on calcule donc la diagonale de cette forme, C est a dire la plus grande distance entre deux carrés
    """
    carre1 = listeCoordCarre[0]
    carre2 = listeCoordCarre[1]
    carre3 = listeCoordCarre[2]
    carre4 = listeCoordCarre[3]
    
    distance1 = size_carre((carre1,carre2))
    distance2 = size_carre((carre1,carre3))
    distance3 = size_carre((carre1,carre4))
    distanceMax = max(distance1,distance2,distance3)
    return distanceMax

def translation_image(image, coord, coordReference):

    """
    déplace une image pour qu'un motif qui apparait a une coordonnée coord, se retrouve a une coordonée
    coordReference
    """
    x,y = coord
    x_standard, y_standard = coordReference
    decalage_x = x_standard - x
    decalage_y = y_standard - y

    rows, cols = image.shape
    M = np.float32([[1,0,decalage_x],[0,1,decalage_y]])
    imageTranslate = cv2.warpAffine(image,M,(cols,rows))
    return imageTranslate

def agrandissement_image(image,taille,tailleReference):

    """agrandi l'image pour qu'un segment de taille: taille ,
    deviennent un segment de taille : tailleReference

    """
    facteurAgrandissement = tailleReference/taille
    tailleOriginal = image.shape
    x_agrandi = math.floor(facteurAgrandissement * tailleOriginal[0])
    y_agrandi = math.floor(facteurAgrandissement * tailleOriginal[1])
    tailleAgrandi = (y_agrandi, x_agrandi)
    imageAgrandi = cv2.resize(image, tailleAgrandi)
    return imageAgrandi

def processing_cadrage(image):

    equalized = cv2.equalizeHist(image)
    ret, seuil = cv2.threshold(equalized, 12, 255, cv2.THRESH_BINARY)
   
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE, kernel, iterations = 1) #erosion suivi d'une dilatation
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 6) #erosion suivi d'une dilatation
    dilate = cv2.dilate(opening,kernel,iterations = 2)
    erosion = cv2.erode(dilate,kernel,iterations = 2)
    return erosion


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


##########          Chargement de l'image           #######################

fichier = filename
imageOriginal = cv2.imread(fichier)


#########         Pre-processing                    #######################

gray = cv2.cvtColor(imageOriginal, cv2.COLOR_BGR2GRAY) #passage en nuance de gris

#unsharp masking
flou = cv2.GaussianBlur(gray,(9,9),10)   
mask = gray - flou
sharp = cv2.addWeighted(gray, 1, mask , 0.03, 0) #ajoute l'original avec le mask pondéré


#laplacian shapening
laplacien = cv2.Laplacian(sharp,cv2.CV_64F)
laplacienScaled = cv2.convertScaleAbs(laplacien)
image = cv2.addWeighted(sharp, 1, laplacienScaled , -0.2, 0)

#changer a true pour afficher le spectre frequenciel
if(False ):
    #transformation en fréquence
    dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)      #transforme en fréquence
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    #filtrage des frequences
    rows, cols = image.shape
    crow,ccol = math.floor(rows/2) , math.floor(cols/2)
    mask = np.ones((rows,cols,2),np.uint8)
    mask[crow-500:crow+500, ccol-100:ccol+100] = 0
    fshift = dft_shift*mask
    magnitude_spectrum2 = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))

    #reconstruction de l'image
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    #affichage du spectre
    plt.subplot(141),plt.imshow(image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(143),plt.imshow(magnitude_spectrum2, cmap = 'gray')
    plt.title('Magnitude Spectrum Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.show()

imageTraitee = image


##########        Recadrage de l'image           #######################

original = cv2.imread(fichier)

#pre-processing image
#   fait les taches de pre processing pour cadrer l'image
#   met en nuance de gris, puis threshold, puis morphologie

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
imageProcessed = processing_cadrage(gray)

#recupere les coordonnées des carrés
listeCarres = detect_carre(imageProcessed)

#si moins de 4, on ne peux pas recadrer l'image
if(len(listeCarres) >= 4):
    listeBonCarres = select_bon_carre(listeCarres)

    #calcule la taille de la forme regroupant les 4 carrées
    taille = detect_taille_motif(listeBonCarres)

    #distance mesure directemnt sur 1 image de reference=> 765 pixel de diagonale
    tailleReference = 765

    #agrandissement du document
    imageAgrandi = agrandissement_image(imageTraitee,taille,tailleReference)

    #repetions des etapes precedentes avec une image agrandie
    imageProcessed = processing_cadrage(imageAgrandi)
    listeCarres = detect_carre(imageProcessed)
    listeBonCarres = select_bon_carre(listeCarres)

    #calcule le centre de la forme regroupant les 4 carrés
    centre = detect_centre_motif(listeBonCarres)

    #centre mesurée directement sur l'image de reference=> (626,350)
    centreReference = (626,350)

    #translation du document
    imageTranslated = translation_image(imageAgrandi, centre, centreReference)
    imageTraitee = imageTranslated
    
    """
    print("trouvé au moins 4 carrés !")
    print("taille du motif avant grossisment : " + str(taille))
    print("centre du motif avant grossisment : " + str(centre))
    print("resolution apres agrandissemnt : " + str(imageAgrandi.shape))
     """
"""
#show the squares detected
plt.imshow(imageProcessed, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
"""

#print(listeCarres) #affiche dans le terminal les coordonées des carrés pour une verification manuelle


#cv2.imshow('image traité',imageTraitee)

##########        Recuperation des champs           #######################
#les coordonnées sont sous la forme (gauche,haut,droite,bas)


#ID carte
coord = (220,98,426,126)
numeroID = delimited_ocr(imageTraitee,coord)

#numero carte       
coord = (49,175,193,197)
numero = delimited_ocr(imageTraitee,coord)

#immatriculation  (A)
coord = (236,175,424,197)
immatriculation = delimited_ocr(imageTraitee,coord)

#date premiere d'immatriculation  (B)
coord = (57,218,414,260)
datePremiereImmatriculation = delimited_ocr(imageTraitee,coord)

#nom proprietaire        (C.2.1)
coord = (97,263,415,293)
nomProprietaire = delimited_ocr(imageTraitee,coord)

#adresse proprietaire       (C.2.3)      
coord = (99,300,414,366)
adresseProprietaire = delimited_ocr(imageTraitee,coord)

#marque vehicule       (D.1)
coord = (507,57,830,85)
marqueVehicule = delimited_ocr(imageTraitee,coord)

#type vehicule       (D.2)
coord = (507,93,830,127)
typeVehicule = delimited_ocr(imageTraitee,coord)

#nom commercial       (D.3)
coord = (507,135,830,160)
nomCommercial = delimited_ocr(imageTraitee,coord)

#numero de serie       (E)
coord = (491,167,830,193)
numeroSerie = delimited_ocr(imageTraitee,coord)

#masse maximale         (F.1)
coord = (507,195,830,211)
masseMax = delimited_ocr(imageTraitee,coord)

#masse maximale 2       (F.2)
coord = (507,207,830,230)
masseMax2 = delimited_ocr(imageTraitee,coord)

#date premiere d'immatriculation  (I)
coord = (491,226,830,249)
dateDerniereImmatriculation = delimited_ocr(imageTraitee,coord)

#categorie vehicule   (J)
coord = (491,247,830,265)
categorieVehicule = delimited_ocr(imageTraitee,coord)

#affiche_zone(imageTraitee,coord)
cv2.imshow('image final',imageTraitee)

#Affichage résultat
if(False):
    print("champ 1: " + numeroID)
    print("champ 2: " + numero)
    print("champ A: " + immatriculation)
    print("champ B: " + datePremiereImmatriculation)
    print("champ C.2.1: " + nomProprietaire)
    print("champ C.2.3: " + adresseProprietaire)
    print("champ D.1: " + marqueVehicule)
    print("champ D.2: " + typeVehicule)
    print("champ D.3: " + nomCommercial)
    print("champ E: " + numeroSerie)
    print("champ F.1: " + masseMax)
    print("champ F.2: " + masseMax2)
    print("champ I: " + dateDerniereImmatriculation)
    print("champ J: " + categorieVehicule)


#####################     CREATION DE LA STRUCTURE XML          ############

path = "generatedXML.xml"

carteGrise = Element('carteGrise')

### DONNEES IMMATRICULATION

XMLdonneesImmatriculation = SubElement(carteGrise,'donneesImmatriculation')
XMLnumeroImmatriculation = SubElement(XMLdonneesImmatriculation, 'numeroImmatriculation')
XMLdatePremiereImmatriculation = SubElement(XMLdonneesImmatriculation,'datePremiereImmatriculation')
XMLdateImmatriculationCourante = SubElement(XMLdonneesImmatriculation, 'dateImmatriculationCourante')

XMLdureeValidite = SubElement(XMLdonneesImmatriculation, 'dureeValidite')

### DONNEES NOMINATIVE

XMLdonneesNominative = SubElement(carteGrise,'donneesNominative')
XMLproprietaireCarteGrise =SubElement(XMLdonneesNominative, 'proprietaireCarteGrise')
XMLnom1 = SubElement(XMLproprietaireCarteGrise, 'nom')
XMLpenom1 = SubElement(XMLproprietaireCarteGrise, 'prenom')
XMLdateN1 = SubElement(XMLproprietaireCarteGrise, 'dateNaissance')
XMLadresse1 = SubElement(XMLproprietaireCarteGrise, 'adresse')

XMLproprietaireVehicule =SubElement(XMLdonneesNominative, 'proprietaireVehicule')
XMLnom2 = SubElement(XMLproprietaireVehicule, 'nom')
XMLpenom2 = SubElement(XMLproprietaireVehicule, 'prenom')
XMLdateN2 = SubElement(XMLproprietaireVehicule, 'dateNaissance')
XMLadresse2 = SubElement(XMLproprietaireVehicule, 'adresse')

XMLutilisateurVehicule =SubElement(XMLdonneesNominative, 'utilisateurVehicule')
XMLnom3 = SubElement(XMLutilisateurVehicule, 'nom')
XMLpenom3 = SubElement(XMLutilisateurVehicule, 'prenom')
XMLdateN3 = SubElement(XMLutilisateurVehicule, 'dateNaissance')
XMLadresse3 = SubElement(XMLutilisateurVehicule, 'adresse')


#DONNEES VEHICULES

XMLdonneesVehicules = SubElement (carteGrise, 'informationVehicule')
XMLinformationsGenerales = SubElement (XMLdonneesVehicules, 'informationsGenerales')
XMLmarque = SubElement(XMLinformationsGenerales, 'marque')
XMLtypeVehicule = SubElement(XMLinformationsGenerales,'typeVehicule')
XMLnomCommercial = SubElement(XMLinformationsGenerales, 'denominationCommerciale')
XMLcouleur = SubElement(XMLinformationsGenerales, 'couleur')
XMLplaces = SubElement(XMLinformationsGenerales, 'places')
XMLplaceAssise = SubElement(XMLplaces, 'placeAssise')
XMLplaceDebout = SubElement(XMLplaces, 'placeDebout')
XMLcategorieVehicule = SubElement(XMLinformationsGenerales, 'categorieVehicule')
XMLdestinationEtUsage = SubElement(XMLinformationsGenerales, 'destinationEtUsage')
XMLcarroserie = SubElement(XMLinformationsGenerales, 'carroserie')


#DONNEES TECHNIQUES
XMLinformationTechniques = SubElement(XMLdonneesVehicules,'informationsTechniques')

#DONNES ESSENTIELLES
XMLdonneesEssentielles = SubElement(XMLinformationTechniques,'donneesEssentielles')
XMLnumeroSerie = SubElement(XMLdonneesEssentielles,'numeroSerie')
XMLnumeroHomologation = SubElement(XMLdonneesEssentielles,'numeroHomologation')
XMLaxeData = SubElement(XMLdonneesEssentielles,'axeData')
XMLnombreAxe = SubElement(XMLaxeData,'nombreAxe')
XMLdistanceInterAxe = SubElement(XMLaxeData,'distanceInterAxe')
XMLrapportPuissanceMasse = SubElement(XMLdonneesEssentielles,'rapportPuissanceMasse')
XMLvitesseMax = SubElement(XMLdonneesEssentielles,'vitesseMax')
XMLcapaciteReservoir = SubElement(XMLdonneesEssentielles,'capaciteReservoir')

#MOTEUR
XMLmoteur = SubElement(XMLinformationTechniques,'moteur')
XMLcylindree = SubElement(XMLmoteur,'cylindree')
XMLpuissanceMaximale = SubElement(XMLmoteur,'puissanceMaximale')
XMLtypeCarburant = SubElement(XMLmoteur,'typeCarburant')
XMLregimeNominal = SubElement(XMLmoteur,'regimeNominal')
XMLtypeMoteur = SubElement(XMLmoteur,'typeMoteur')

#ENVIRONEMENTAL
XMLenvironnemental = SubElement(XMLinformationTechniques,'environnemental')
XMLniveauxSonores = SubElement(XMLenvironnemental,'niveauxSonores')
XMLarret = SubElement(XMLniveauxSonores,'arret')
XMLmarche = SubElement(XMLniveauxSonores,'marche') 
XMLemissionGaz = SubElement(XMLenvironnemental,'emissionGaz')
XMLCO2 = SubElement(XMLemissionGaz,'CO2')
XMLclasseEnvironnementaleCE = SubElement(XMLemissionGaz,'classeEnvironnementaleCE')

#MASSE
    
XMLmasses = SubElement(XMLinformationTechniques,'masses')
XMLmaxTechniquementAdmissible = SubElement(XMLmasses,'maxTechniquementAdmissible')
XMLmaxVehiculeEnService = SubElement(XMLmasses,'maxVehiculeEnService')
XMLmaxEnsembleEnService = SubElement(XMLmasses,'maxEnsembleEnService')
XMLmasseEnService = SubElement(XMLmasses,'masseEnService')
XMLremorques = SubElement(XMLmasses,'remorques')
XMLfreinee = SubElement(XMLremorques,'freinee')
XMLnonFreinee = SubElement(XMLremorques,'nonFreinee')
XMLrepartitionAxes = SubElement(XMLmasses,'repartitionAxes')
XMLaxe = SubElement(XMLrepartitionAxes,'axe')
XMLnumeroAxe = SubElement(XMLaxe,'numeroAxe')
XMLmasseAxe = SubElement(XMLaxe,'masseAxe')

#Attribution des valuers
XMLmarque.text = marqueVehicule
XMLnumeroSerie.text = numeroSerie
XMLtypeVehicule.text = typeVehicule
XMLnomCommercial.text = nomCommercial
XMLcategorieVehicule.text = categorieVehicule
XMLadresse2.text = adresseProprietaire
XMLnom2.text = nomProprietaire
XMLdatePremiereImmatriculation.text = datePremiereImmatriculation
XMLdateImmatriculationCourante.text = dateDerniereImmatriculation
XMLnumeroImmatriculation.text = immatriculation
XMLmaxTechniquementAdmissible.text = masseMax
XMLmaxVehiculeEnService.text = masseMax2





####################             GENERATION DU XML            #######################

if (os.path.exists(path)):
    os.remove(path)
f= open(path,mode = 'xb')

#prettify permet de rendre l'xml formatte pour la lecture humaine
buffer = bytes(prettify(carteGrise),'utf-8')
f.write(buffer)
f.close()




















