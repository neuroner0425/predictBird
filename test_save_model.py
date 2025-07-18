# 학습한 모델이 잘 작동되는지 확인하는 코드

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from PIL import Image

# 아래는 val 데이터 테스트해서 정확도 얼마나 나오는지 확인하는 함수
def test_val(_classes, _model, _transform):
    data_dir = 'resources/train'
    batch_size = 32

    dataset = datasets.ImageFolder(data_dir, transform=_transform)
    dataset.classes = _classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(_classes)}

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])  # 학습 데이터 분리는 필요없음

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = _model(xb)
            _, predicted = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    print(f'Validation Accuracy: {correct / total:.4f}')

# 아래 두개는 하나의 이미지를 테스트해보는 함수
def preprocess_image(_img_path, _transform):
    img = Image.open(_img_path).convert('RGB')
    img_tensor = _transform(img).unsqueeze(0)
    return img_tensor

def predict_topk(_model, _img_tensor, _classes, _device, k=5):
    _model.eval()
    with torch.no_grad():
        _img_tensor = _img_tensor.to(_device)
        outputs = _model(_img_tensor)
        probs = F.softmax(outputs, dim=1)
        topk_probs, topk_idxs = probs.topk(k, dim=1)
        topk_probs = topk_probs.cpu().numpy().flatten()
        topk_idxs = topk_idxs.cpu().numpy().flatten()
        print('Top-5 예측 결과:')
        for i in range(k):
            print(f'{i+1}: {_classes[topk_idxs[i]]} ({topk_probs[i] * 100:.2f}%)')

# 메인 함수
if __name__ == '__main__':
    classes = ['Abraxas latifasciata', 'Abraxas niphonibia', 'Acanthosoma forficula', 'Acrida cinerea', 'Actias gnoma',
               'Adoretus tenuimaculatus', 'Agapanthia pilicornis', 'Agelastica coerulea', 'Agnidra scabiosa',
               'Agrypnus argillaceus', 'Agylla gigantea', 'Aiolocaria hexaspilota', 'Algon sphaericollis',
               'Allomyrina dichotoma', 'Amata germana', 'Ammophila campestris', 'Anatis halonis', 'Anax parthenope',
               'Angerona prunaria', 'Anoplocnemis dallasi', 'Anoplophora malasiaca', 'Anotogaster sieboldii',
               'Anthinobaris dispilota', 'Anthocharis scolymus', 'Anthrax distigma', 'Anthrax jezoensis',
               'Apatura iris', 'Apatura metis', 'Apis cerana', 'Apis mellifera', 'Aporia crataegi', 'Appasus japonicus',
               'Apriona germari', 'Aquarius paludum', 'Argyreus hyperbius', 'Aromia bungii', 'Asias halodendri',
               'Atractomorpha lata', 'Atuphora stictica', 'Baculum elongatum', 'Baliga micans', 'Batocera lineolata',
               'Bombus ignitus', 'Bombylius major', 'Bothrogonia japonica', 'Brahmaea certhia',
               'Callambulyx tatarinovi', 'Callambulyx tatarinovii', 'Callipogon relictus', 'Callygris compositata',
               'Calopteryx atrata', 'Calopteryx japonica', 'Campalita chinense', 'Carabus smaragdinus',
               'Carabus sternbergi', 'Carpocoris purpureipennis', 'Cerambycidae', 'Ceriagrion melanurum',
               'Chalicodoma sculpturalis', 'Chelonomorpha japona', 'Chloridolum (Chloridolum) sieversi',
               'Chlorophanus grandis', 'Chromogeotrupes auratus', 'Chrysochroa fulgidissima', 'Chrysochus chinensis',
               'Chrysomela populi', 'Chrysomela vigintipunctata', 'Cicindela chinensis', 'Cicindela sachalinensis',
               'Coccinella septempunctata', 'Coccinellidae Latreille', 'Coenagrion ecornutum', 'Colias erate',
               'Conocephalus (Anisoptera) exemptus', 'Conocephalus chinensis', 'Cophinopoda chinensis',
               'Copris tripartitus', 'Coptosoma bifarium', 'Corymbia rubra', 'Crambus perlellus',
               'Crocothemis servilia', 'Cryptotympana atrata', 'Cucujus coccinatus Lewis', 'Curculio sikkimensis',
               'Cybister brevis', 'Cybister japonicus', 'Cyntia cardui', 'Daimio tethys', 'Damaster jankowskii',
               'Dicranocephalus adamsi', 'Dictyophara patruelis', 'Dilipa fenestra', 'Dorcus hopei', 'Ducetia japonica',
               'Eoscartopsis assimilis', 'Epicauta chinensis', 'Epicopeia menciana', 'Epipomponia nawai',
               'Episomus turritus', 'Episyrphus balteatus', 'Eristalis cerealis', 'Erynnis montanus', 'Eumenis autonoe',
               'Euricania facialis', 'Eurydema gebleri', 'Everes argiades', 'Evonymus mandschurian',
               'Fabriciana nerippe', 'Forficula scudderi', 'Gampsocleis sedakovi', 'Gandaritis fixseni',
               'Gastrimargus marmoratus', 'Geisha distinctissima', 'Gonepteryx rhamni', 'Graphosoma rubrolineatum',
               'Graptopsaltria nigrofuscata', 'Gryllotalpa orientalis', 'Hestina assimilis', 'Hexacentrus unicolor',
               'Holotrichia parallela', 'Hydrophilus acuminatus', 'Ischnura asiatica', 'Ivela auripes',
               'Japonica lutea', 'Japonica saepestriata', 'Laccotrephes japonensis', 'Lamiomimus gottschei',
               'Leptidea amurensis', 'Leptosemia takanonis', 'Lestes sponsa', 'Lethocerus deyrolli',
               'Libelloides sibiricus', 'Libythea celtis', 'Limois emelianovi', 'Linaeidea aenea', 'Lixus maculatus',
               'Locusta migratoria', 'Lucanus maculifemoratus', 'Lucilia caesar', 'Luciola lateralis',
               'Luehdorfia puziloi', 'Lycaena dispa', 'Lycaena phlaeas', 'Lychnuris rufa', 'Lycorma delicatula',
               'Lygaeus equestris', 'Lymantria mathura', 'Lyriothemis pachygastra', 'Macrodorcas recta',
               'Macroglossum pyrrhostictum', 'Maladera ovatula', 'Mantis religiosa', 'Mecorhis ursulus',
               'Megachile nipponica', 'Megopis sinica', 'Meimuna mongolica', 'Meimuna opalifera',
               'Meloe violaceus semenowi', 'Meloimorpha japonica', 'Melolontha incana', 'Metopta rectifasciata',
               'Metrioptera bonneti', 'Micadina phluctainoides', 'Mimathyma schrenckii', 'Mimela splendens',
               'Minois dryas', 'Moechotypa diphysis', 'Molipteryx fuliginosa', 'Monochamus alternatus',
               'Nannophya pygmaea', 'Nemophora staudingerella', 'Neptis pryeri', 'Neptis rivularis', 'Neptis sappho',
               'Neuroctenus castaneus', 'Nezara antennata', 'Nicrophorus concolor', 'Notonecta triguttata',
               'Ochlodes subhyalina', 'Ochlodes venata', 'Oedaleus infernalis', 'Oides decempunctatus',
               'Olenecamptus octopustulatus', 'Oncotympana fuscata', 'Orancistrocerus drewseni',
               'Orchestes sanguinipes', 'Oreumenes decoratus', 'Ornatalcides trifidus', 'Orthetrum albistylum',
               'Orthetrum japonicum', 'Oxya chinensis', 'Palomena angulosa', 'Panorpa coreana', 'Papilio bianor',
               'Paracercion calamorum', 'Paracycnotrachelus longiceps', 'Parantica sita', 'Parapolybia crocea',
               'Parapolybia varia', 'Paratlanticus ussuriensis', 'Parnara guttata', 'Parnassius stubbendorfii',
               'Patanga japonica', 'Pectocera fortunei', 'Pedicia daimio', 'Pentatoma japonica',
               'Pentatoma metallifera', 'Pentatoma parametallifera', 'Pericallia matronula', 'Phaneroptera falcata',
               'Phaneroptera nigroantennata', 'Pheropsophus jessoensis', 'Philaronia nigrifrons', 'Phraortes elongatus',
               'Pieris melete', 'Pieris rapae', 'Placosternum esakii', 'Platycnemis phyllopoda',
               'Platypleura kaempferi', 'Plesiophthalmus davidis', 'Podabrus dilaticollis', 'Poecilocoris lewisi',
               'Poecilocoris splendidulus', 'Polistes jokahamae', 'Polygonia c-aureum', 'Popillia flavosellata',
               'Prionus insularis', 'Problepsis superans', 'Promachus yesonicus', 'Prosopocoilus blanchardi',
               'Prosopocoilus inclinatus', 'Prosthiochaeta bifasciata', 'Pseudopyrochroa rufula', 'Pseudothemis zonata',
               'Pseudotorynorrhina japonica', 'Purpuricenus lituratus', 'Quedius (Velleius) pectinatus',
               'Ranatra chinensis', 'Ranatra unicolor', 'Rapala caerulea', 'Rhodinia fugax', 'Rhyothemis fuliginosa',
               'Ricania taeniata', 'Riptortus clavatus', 'Ruspolia lineosa', 'Sarcophaga melanura', 'Sasakia charonda',
               'Sastragala esakii', 'Scathophaga stercoraria', 'Scintillatrix pretiosa', 'Sephisa princeps',
               'Sericinus montela', 'Serrognathus platymelus', 'Shirakiacris shirakii', 'Sieboldius albardae',
               'Silpha perforata', 'Sipalinus gigas', 'Sphedanolestes impressicollis', 'Sphragifera biplagiata',
               'Spirama retorta', 'Spoladea recurvalis', 'Statilia maculata', 'Stethophyma magister',
               'Sympecma paedisca', 'Sympetrum depressiusculum', 'Sympetrum infuscatum', 'Sympetrum pedemontanum',
               'Syrphidae', 'Tabanus chrysurus', 'Tachina jakovlevi', 'Tachycines coreanus', 'Teleogryllus emma',
               'Tenodera angustipennis', 'Tenodera aridifolia', 'Tetrix japonica', 'Tettigetta isshikii',
               'Tettigonia viridissima', 'Thyris fenestrella', 'Timandra comptaria', 'Timomenus komarowi',
               'Tipula aino Alexander', 'Tongeia fischeri', 'Trichius succinctus', 'Trigomphus citimus',
               'Urochela quadrinotata', 'Velarifictorus aspersus', 'Vespa crabro', 'Vespa ducali', 'Vespula flaviceps',
               'Volucella pellucens', 'Xylocopa appendiculata']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = models.resnet50(weights=None)  # 사전학습 weight는 필요 없음
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()

    # 여기 위까지는 모델 불러오는 코드
    # 아래는 어떻게 테스트 해 볼것인지
    여러개_테스트 = 1

    if(여러개_테스트):
        test_val(classes, model, transform)
    else:
        test_img_path = 'resources/test.png'  # ← 테스트할 이미지 경로
        img_tensor = preprocess_image(test_img_path, transform)
        predict_topk(model, img_tensor, classes, device, k=5)