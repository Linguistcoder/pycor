# todo: convert labels to int
# todo: mapping between DanNet senses and int

key = {}

# 1-5 cor-s
key['ansigt-1'] = 1
key['ansigt-1b'] = 2
key['ansigt-1a'] = 3
key['ansigt-3'] = 5
key['ansigt-3a'] = 5
key['ansigt-2'] = 4
key['ansigt-2a'] = 4

# 6-14 fixed expressions
key['ansigt-F-en-våd-klud-i-ansigtet'] = 6
key['ansigt-F-lige-op-i-ansigtet/lige-op-i-éns-åbne-ansigt'] = 7
key['ansigt-F-med-et-menneskeligt-ansigt'] = 8
key['ansigt-F-skære-ansigt/ansigter/lave-ansigt/ansigter'] = 9
key['ansigt-F-slag-i-ansigtet'] = 10
key['ansigt-F-stå-ansigt-til-ansigt'] = 11
key['ansigt-F-stå-malet-i-ansigtet-på-nogen'] = 12
key['ansigt-F-sætte-ansigt-på'] = 13
key['ansigt-F-tabe/miste-ansigt'] = 14

key['blik-1-4a'] = 3
key['blik-1-1'] = 1
key['blik-2-1'] = 4
key['blik-1-2'] = 1
key['blik-1-3'] = 2
key['blik-1-4'] = 3
key['blik-F-få-blikket'] = 5
key['blik-F-indre-blik'] = 6

key['hold-1a'] = '5579'
key['hold-1b'] = '5579'
key['hold-1'] = '5579'
key['hold-2'] = '8265'
key['hold-3'] = '17039'
key['hold-4'] = '40436'
key['hold_5'] = '7823'
key['hold-5a'] = '22237'
key['hold_6'] = '75192'
key['hold-6a'] = '71987'
key['hold-8'] = '67127'
key['hold-F-på-hold'] = '70927'
key['hold_F_være/spille med på holdet'] = '24050'

key['hul-1a'] = '58347'  # åbent sår eller brud i huden eller et andet st...	natural+object+bodypart
key['hul-1'] = '58347'
key['hul-1b'] = '58347'
key[
    'hul-1-1b-1e'] = '2270'  # Manden havde hul i bukserne og store huller i skoene #	gennembrudt, udgravet eller på anden måde opst...	place+object
key['hul-1c'] = '31873'  # hver af de små fordybninger der findes på en g...	place+object+artifact
key[
    'hul-1d'] = '18490'  # FRA {tomrum_1_2}	frit, uudnyttet område el. tidsrum (Brug: ""Ce...	3rdOrderEntity+Mental	OVERFØRT noget der er tomt (og umuligt at udfy...	ikke i DanNet (ikke i download i hvert fald)
key['hul-1f'] = '58364'  # OVERFØRT lille, indelukket og mørkt rum SPROGB...	building+object+part
key['hul-2'] = '58378'  # tomt mellemrum i en række eller kreds	place+object+part
key['hul-3'] = '18260'  # sygdomsangrebet område i en tand, opstået ved ...	natural+object+bodypart
key['hul-4'] = '58363'  # naturlig åbning i et menneskes krop ; den kvin...	natural+object+bodypart
key['hul-2a'] = '20793'  # OVERFØRT område der mangler at være dækket ind...	3rdOrder+mental
key['hul-2b'] = '57221'  # OVERFØRT økonomisk udgift eller gæld som der m...	3rdOrder+quantity
key['hul-2c'] = '74662'  # OVERFØRT tomrum i et forløb, fx en tidsperiode...	3rdOrder+time
key['hul-F-få-hul-på-noget'] = '2270'
key['hul-F-gå-hul'] = '2270'
key['hul-F-hul-igennem'] = '2270'
key['hul-F-hul-i-hovedet'] = '2270'
key['hul-F-hul-i-jorden'] = '2270'
key['hul-F-hul-på-bylden'] = '2270'
key['hul-F-skide-hul-i'] = '2270'
key['hul-F-sort-hul'] = '37519'
key['hul-F-tage-hul-på-noget'] = '67474'

key['kort-1'] = '20659'
key[
    'kort-1c'] = '20659'  # det er .. ikke pengeinstitutterne, der afgør, hvornår en kunde skal hæfte for en tyvs misbrug af kort og kode
key['kort-1b'] = '25176'  # "Hvor mange gange skal jeg klippe på et gult kort til Hillerød." "To klip"
key[
    'kort-1a'] = '25182'  # Fra brevkort: Da jeg samler på danske postkort, vil jeg blive meget glad, hvis [nogen] vil sende mig nogle kort
key['kort-1e'] = '25201'  # kort som dokumenterer at en person er den vedkommende giver sig ud for
key['kort-3'] = '25197'  # Der spilles stadig kort .. Men nu spiller vi kun om øl. Ikke penge SeHør1983
key['kort-2a'] = '25238'  # i fremtiden vil [der] blive lavet kort over alle menneskers arveanlæg
key['kort-2'] = '25238'  # Bogen indeholdt et geodætisk kort over Danmark .. og kort over 200 danske byer
key['kort-4'] = '25269'  # [han har] bygget en komplet 16-bit datamat med 91 integrerede kredsløb på et enkelt kort
key['kort-F-lægge-kortene-på-bordet'] = '32121'
key['kort-F-gult-kort'] = '25202'
key['kort-F-røde-kort'] = '25203'
key['kort-F-blande-kortene'] = '25197'
key['kort-F-gult-kort-overført'] = '25202'
key['kort-F-have-gode-kort-på-hånden'] = '25197'
key['kort-F-holde-kortene-tæt-ind-til-kroppen'] = '25197'
key['kort-F-kigge-nogen-i-kortene'] = '25197'
key['kort-F-ligge-i-kortene'] = '25197'
key['kort-F-røde-kort-overført'] = '25203'
key['kort-F-spille-med-åbne-kort'] = '25197'
key['kort-F-spille-sine-kort-godt/dårligt'] = '25197'
key['Kort_F_sætte alt på et kort'] = '25197'

key['lys-1'] = '29405'
key['lys-1a'] = '37263'  # fra stråling_1
key['lys-1b'] = '29391'  # Fra dagslys_1
key['lys-1c'] = '58462'  # Fra lampelys_0
key['lys-1h'] = '47346'
key[
    'lys-2'] = '73014'  # Fra rampelys_2: Scenografen arbejder naturligvis tæt sammen med teknikerne. At sætte lys til en teaterforestilling er et helt fagområde for sig
key['lys-1d'] = '18340'
key['lys-1e'] = '78536'
key['lys-1f'] = '78537'
key['lys-1g'] = '14410'
key['lys-3'] = '24870'
key['lys-4'] = '5788'
key['lys-5'] = '7306'
key['lys-F-der-går-et-lys-op-for-nogen'] = '78536'
key['lys-F-føre-nogen-bag-lyset'] = '29405'
key['lys-F-grønt/gult/rødt-lys'] = '47363'
key['lys-F-grønt-lys(for-noget)'] = '47363'
key['lys-F-gå-ud-som-et-lys'] = '74719'
key['lys-F-i-et-nyt-(andet,-..)-lys'] = '29405'
key['lys-F-i-lyset-af'] = '29405'
key['lys-F-kaste-lys-over'] = '29405'
key['lys-F-komme-for-dagens-lys'] = '29405'
key['lys-F-lede/søge-med-lys-og-lygte'] = '5788'
key['lys_F_levende lys'] = '24870'
key['lys-F-lys-i-mørket'] = '14411'
key['lys-F-komme/bringe-frem-i-lyset'] = '29405'
key['lys-F-se-lyset'] = '78536'
key['lys-F-stille/sætte-i-et-dårligt-lys'] = '29405'
key['lys-F-tåle-dagens-lys'] = '29405'
key['lys-F-sætte-sit-lys-under-en-skæppe'] = '24870'
key['lys-F-se-dagens-lys'] = '29405'

key['model-1'] = '42152'
key['model-1a'] = '7443'
key['model-2'] = '42153'
key['model-3'] = '15683'
key['model-4'] = '42150'
key['model-5'] = '7433'
key['model-6'] = '7438'
key['model-6a'] = '7449'
key['model-F-stå-model-til'] = '7433'

key['plade-1'] = '286'
key['plade-1a'] = '14135'
key['plade-F-panser-og-plade'] = '2707'
key['plade-1b'] = '14534'
key['plade-1c'] = '14535'
key['plade-2'] = '47878'
key['plade-2a'] = '23793'
key['plade-4'] = '58948'
key['plade-F-gule-plader'] = '58947'
key['plade-F-stikke-en-plade'] = '72750'
key['plade-6'] = '18526'
key['plade-5'] = '18591'
key['plade-F-hvide-plader'] = '41540'

key['skade-1-1'] = '37313'
key['skade-2-1'] = '11363'
key[
    'skade-1-1c'] = '41271'  # fra {skadevirkning_1}	skade opstået som følge af noget;" negativ føl... #Satte vi produktionen op .. ville der stå en Morgan på hvert andet gadehjørne til skade for efterspørgslen
key['skade-1-1b'] = '67613'  # hash kan give psykiske skader
key['skade-1-1a'] = '38653'
key['skade-F-det-er-ingen-skade-til'] = '38653'
key[
    'skade-F-(det-er/var)-skade'] = '75089'  # fra {skam,1_3}	beklageligt, ærgerligt el. uheldigt forhold (B...	Property+Mental
key['skade-F-føje-spot-til-skade'] = '38653'
key['skade-F-komme-for-skade'] = '38653'
key['skade-F-komme-til-skade'] = '38653'
key['skade-F-slå-halv-skade'] = '37313'
key['skade-F-tage/lide-skade'] = '38653'

key['slag-4a'] = '37213'  # fra bogstavelig betydning i slag-4
key['slag-4'] = '37213'
key['slag-6b'] = '24091'
key['slag-1'] = '22414'
key['slag-2b'] = '31301'
key['slag-1c'] = '48605'
key['slag-3'] = '17951'
key[
    'slag-6'] = '37293'  # fra terningkast. #tur i spil hvor man kaster en eller flere terninger og evt. rykker sin brik på spillebrættet
key['slag-3a'] = '72503'  # fra synonym pulsslag
key['slag-2c'] = '31301'
key['slag-2'] = '31301'
key['slag-6a'] = '24091'
key['slag-1a'] = '22414'
key['slag-1b'] = '56270'  # {øretæve_1_1}	nederlag; modgang;" kritik (Brug: ""De konserv...
key['slag-2a'] = '37371'
key['slag-7'] = '9545'
key['slag-F-frit-slag'] = '56973'  # {adgang_1_1}	mulighed for el. ret til at få fat i noget el....	3rdOrderEntity
key['slag-F-klokken-falder-i-slag'] = '67203'  # fra klokkeslag
key['slag-F-med-ét-slag'] = '67203'
key['slag-F-hjerte-springer-slag-over'] = '17951'
key['slag-F-på-slaget'] = '67204'
key['slag-F-slag-i-ansigtet'] = '22414'
key['slag-F-slag-i-bolledejen'] = '22414'
key['slag-F-slag-i-luften'] = '22414'
key['slag-F-slag-i-slag'] = '9545'
key['slag-F-slag-på-tasken'] = '27489'  # eksisterede i forvejen
key['slag-F-slå-et-slag'] = '22414'
key['Slag_F_små slag)'] = '22414'
key['slag-F-stor-i-slaget'] = '22414'

key['stykke-5b'] = '15433'
key['stykke-5c'] = '14185'  # afgrænset del af en tekst #fra tekststykke
key['stykke-1'] = '14177'
key['stykke-1a'] = '48798'
key['stykke-2'] = '14178'
key['stykke-2a'] = '35533'  # del af et (dyrket) område. {jordstykke_1}
key['stykke-3'] = '36802'
key['stykke-3a'] = '20970'
key['stykke-4'] = '49018'
key['stykke-4a'] = '6918'
key['stykke-4b'] = '10004'
key['stykke-4c'] = '16018'  # selvstændig, sammenhængende helhed betragtet s... #fra synset "to alen af et stykke
key['stykke-4d'] = '39044'
key['stykke-5'] = '28274'
key['stykke-5a'] = '15432'
key['stykke-5d'] = '38589'

key['stykke-F-en-4-5-(25,-tusind,-..)-stykker'] = '49018'
key['stykke-F-et-langt-stykke-(af-vejen)/et-langt-stykke-hen-ad-vejen'] = '36802'
key['stykke-F-i-stumper-og-stykker'] = '14177'
key['stykke-F-når-det-kommer-til-stykket'] = '36802'
key['stykke-F-i-stykker'] = '14177'
key['stykke-F-slå-i-stykker'] = '14177'
key['top-2a'] = '10016'
key['top-2b'] = '63072'  # fra topledelse
key['top-1a'] = '15794'
key['top-1'] = '38394'
key['top-2'] = '10016'
key['top-3'] = '3010'
key['top-F-fra-top-til-tå'] = '38394'
key['top-F-på-toppen-af'] = '38394'
key['top-F-til-tops'] = '38394'
key['top-F-toppen-af-isbjerget'] = '38394'
key['top-F-toppen-af/på-kransekagen'] = '38394'
key['top-F-tånd/grå-i-toppen'] = '38394'

key['vold1-1a'] = '25330'
key['vold1-1b'] = '25334'
key[
    'vold1-1c'] = '69808'  # OVERFØRT overgreb der krænker en rettighed, ku... . synset fra overgreb_1, fysisk eller psykisk krænkelse af en svagere ...
key['vold1-1d'] = '25349'  # brug af fysisk kraft eller anstrengelse rettet ..
key[
    'vold1-2'] = '45387'  # kontrol eller herredømme som en stærk person e... synset fra {magt_1}	det at have position og midler til at bestemme...	Property
key['vold1-1'] = '25330'
key['vold2-1'] = '37376'
key['vold2-1a'] = '66920'
key['vold-F-gøre/øve-vold-mod/på'] = '25330'
key['vold-F-med-vold-og-magt'] = '45387'

key['B-selskab-1'] = '38195'
key['B-selskab-1a'] = '63014'
key['B-selskab-1b'] = '38195'  # lagt sammen med hovedbetydning
key['B-selskab-1c'] = '6298'
key['B-selskab-2'] = '5685'
key['B-selskab-2a'] = '5686'
key['B-selskab-4'] = '16930'
key['B-selskab-4a'] = '33535'
key['B-selskab-3'] = '1834'
key['B-selskab-5'] = '33534'
key['B-selskab-F-holde-med-selskab'] = '38195'  # fra {selskab_1}	det at være el. foretage sig noget sammen med ..

key[
    'B-kontakt-2'] = '68735'  # fra nærhed 1. #det at to genstande, dele, flader el.lign. er i fysisk forbindelse eller berøring med hinanden
key['B-kontakt-1a'] = '38217'
key['B-kontakt-1d'] = '38217'
key['B-kontakt-3'] = '1880'
key['B-kontakt-3a'] = '48752'
key['B-kontakt-1'] = '38217'
key['B-kontakt-1b'] = '7397'
key['B-kontakt-2a'] = '47379'
key[
    'B-kontakt-1c'] = '42622'  # fra forbindelse 2. #hvis [arrangørerne] ikke selv vil kontakte en foredragsholder, så sørger vi for kontakten

key[
    'stand-1'] = '25631'  # bestemt tilstand eller forfatning som noget (e...	property+physical		phenomenon		bestemt tilstand eller forfatning som noget (e...
key[
    'stand-2'] = '32170'  # en persons stilling i samfundet, beroende på e...	3rdorder+mental+socuial		cognition		en persons stilling i samfundet, beroende på e...
key[
    'stand-3'] = '5703'  # gruppe af personer der har samme erhverv eller...	3 x human+object+group		person		gruppe af personer der har samme erhverv eller...
key[
    'stand-4'] = '18463'  # særligt indrettet, afgrænset område hvorfra de...	place+Object		location		særligt indrettet, afgrænset område hvorfra de...
key['stand-F-få/bringe-i-stand'] = '25631'  #
# key['stand_F_gøre sig i stand']='25631'#
key['stand-F-gøre-(sætte,-lave,-..)-i-stand'] = '25631'
key['stand-F-holde-stand'] = '25631'
key['stand-F-i-stand-til'] = '25631'
key['stand-F-komme-i-stand'] = '25631'
key['stand-F-se-sig-i-stand-til'] = '25631'
key['stand-F-ude-af-stand-til'] = '25631'

key[
    'plads-1'] = '7571'  # sammenhængende areal eller rumfang der er frit...	Place+Object	Plads: dårligt eksempel – skyldes at DDO's bet...	location	masser af plads, ikke plads til	sammenhængende areal eller rumfang der er frit...
key[
    'plads-1a'] = '57014'  # tid, penge eller andre resurser der gør det mu...	Static		state	plads til at gøre noget, give lads til at rejse	tid, penge eller andre resurser der gør det mu...
key[
    'plads-2'] = '5071'  # sted hvor nogen eller noget anbringes eller be...	Place+Object x 2		location	nogets rette plads, nogens plads i et teater, fly	sted hvor nogen eller noget anbringes eller be...
key['plads-2a'] = '5086'
key[
    'plads-2b'] = '32141'  # lagt sammen med plads-7: post, embede "tilhørsforhold for en person til en bestemt gruppe, opnået ved valg, udtagelse el.lign.
key['plads-2c'] = '5071'  # plads i mit hjerte. lagt sammen med plads-2
key['plads-3'] = '7567'
key[
    'plads-3a'] = '7568'  # åbent, ubebygget område, ofte med en bestemt f...	Place+Object x 2		location	udendørs plads, torv	åbent, ubebygget område, ofte med en bestemt f...
key[
    'plads-4'] = '72643'  # placering i en rangorden, et hierarki eller en...	Relation+location + Relation		relation	plads nr. 3, hans plads i hierarkiet	placering i en rangorden, et hierarki eller en...
key['plads-4a'] = '55201'
key[
    'plads-6'] = '55251'  # en persons adgang til noget, fx uddannelse, pa...	3orderentity		abstract	plads på universitetet, ledige pladser	en persons adgang til noget, fx uddannelse, pa...
key['plads-7'] = '32141'  # post; embede	static+relation		relation	en plads som vicepræsident	post; embede
key['plads-F-banke/sætte-på-plads'] = '5071'  #
key['plads-F-brikkerne-falder-på-plads'] = '5071'  #
key['plads-F-en-plads-i-solen'] = '7571'  #
key['plads-F-på-plads'] = '5071'
key['plads-F-på-pladserne'] = '5071'
key['plads-F-sætte-tingene-på-plads'] = '5071'
key['plads-F-tage-plads'] = '7571'
key['plads-F-vige-pladsen'] = '7571'
key['plads-F-være-på-sin-plads'] = '5071'

alleord = {}

# 16
alleord['ansigt'] = ['ansigt-1', 'ansigt-F-slag-i-ansigtet', 'ansigt-F-lige-op-i-ansigtet/lige-op-i-éns-åbne-ansigt',
                     'ansigt-F-stå-malet-i-ansigtet-på-nogen', 'ansigt-F-skære-ansigt/ansigter/lave-ansigt/ansigter',
                     'ansigt-F-en-våd-klud-i-ansigtet', 'ansigt-F-stå-ansigt-til-ansigt', 'ansigt-2',
                     'ansigt-F-tabe/miste-ansigt', 'ansigt-1b', 'ansigt-2a', 'ansigt-3', 'ansigt-F-sætte-ansigt-på',
                     'ansigt-1a', 'ansigt-3a', 'ansigt-F-med-et-menneskeligt-ansigt']
# 8
alleord['blik'] = ['blik-F-indre-blik', 'blik-1-4a', 'blik-2-1', 'blik-1-2', 'blik-1-4', 'blik-1-1',
                   'blik-F-få-blikket', 'blik-1-3']
# 10
alleord['hold'] = ['hold-8', 'hold-3', 'hold-6a', 'hold-5a', 'hold-1a', 'hold-1b', 'hold-F-på-hold', 'hold-1', 'hold-4',
                   'hold-2']
# 22
alleord['hul'] = ['hul-4', 'hul-2c', 'hul-F-skide-hul-i', 'hul-F-sort-hul', 'hul-2a', 'hul-3', 'hul-1c', 'hul-1d',
                  'hul-1f', 'hul-1', 'hul-F-hul-i-jorden', 'hul-1-1b-1e', 'hul-2', 'hul-F-tage-hul-på-noget',
                  'hul-F-hul-igennem', 'hul-F-få-hul-på-noget', 'hul-F-gå-hul', 'hul-1a', 'hul-2b', 'hul-1b',
                  'hul-F-hul-på-bylden', 'hul-F-hul-i-hovedet']
# 21
alleord['kort'] = ['kort-F-røde-kort-overført', 'kort-F-have-gode-kort-på-hånden', 'kort-F-røde-kort', 'kort-1a',
                   'kort-F-blande-kortene', 'kort-4', 'kort-F-gult-kort-overført',
                   'kort-F-spille-sine-kort-godt/dårligt', 'kort-F-kigge-nogen-i-kortene', 'kort-1c', 'kort-1',
                   'kort-F-ligge-i-kortene', 'kort-2a', 'kort-2', 'kort-F-holde-kortene-tæt-ind-til-kroppen',
                   'kort-F-gult-kort', 'kort-1b', 'kort-1e', 'kort-F-spille-med-åbne-kort', 'kort-3',
                   'kort-F-lægge-kortene-på-bordet']
# 30
alleord['lys'] = ['lys-1f', 'lys-F-føre-nogen-bag-lyset', 'lys-F-kaste-lys-over', 'lys-F-grønt-lys(for-noget)',
                  'lys-F-tåle-dagens-lys', 'lys-F-lede/søge-med-lys-og-lygte', 'lys-F-lys-i-mørket', 'lys-4', 'lys-1c',
                  'lys-F-der-går-et-lys-op-for-nogen', 'lys-F-i-lyset-af', 'lys-5', 'lys-2', 'lys-1e', 'lys-1g',
                  'lys-1h', 'lys-1b', 'lys-F-grønt/gult/rødt-lys', 'lys-F-komme-for-dagens-lys', 'lys-F-se-dagens-lys',
                  'lys-F-se-lyset', 'lys-F-komme/bringe-frem-i-lyset', 'lys-1a', 'lys-F-sætte-sit-lys-under-en-skæppe',
                  'lys-F-i-et-nyt-(andet,-..)-lys', 'lys-F-stille/sætte-i-et-dårligt-lys', 'lys-1',
                  'lys-F-gå-ud-som-et-lys', 'lys-3', 'lys-1d']
# 9
alleord['model'] = ['model-3', 'model-F-stå-model-til', 'model-6', 'model-1', 'model-6a', 'model-5', 'model-4',
                    'model-2', 'model-1a']
# 13
alleord['plade'] = ['plade-5', 'plade-1a', 'plade-F-gule-plader', 'plade-F-panser-og-plade', 'plade-1b',
                    'plade-F-stikke-en-plade', 'plade-F-hvide-plader', 'plade-6', 'plade-1', 'plade-4', 'plade-2',
                    'plade-2a', 'plade-1c']
# 21
alleord['plads'] = ['plads-2c', 'plads-F-banke/sætte-på-plads', 'plads-4', 'plads-F-en-plads-i-solen', 'plads-2b',
                    'plads-3a', 'plads-3', 'plads-4a', 'plads-F-på-pladserne', 'plads-2a', 'plads-2',
                    'plads-F-tage-plads', 'plads-6', 'plads-F-være-på-sin-plads', 'plads-1a', 'plads-1',
                    'plads-F-brikkerne-falder-på-plads', 'plads-F-sætte-tingene-på-plads', 'plads-F-på-plads',
                    'plads-F-vige-pladsen', 'plads-7']
# 12
alleord['skade'] = ['skade-1-1', 'skade-F-(det-er/var)-skade', 'skade-1-1c', 'skade-F-komme-til-skade',
                    'skade-F-komme-for-skade', 'skade-1-1b', 'skade-2-1', 'skade-1-1a',
                    'skade-F-det-er-ingen-skade-til', 'skade-F-tage/lide-skade', 'skade-F-slå-halv-skade',
                    'skade-F-føje-spot-til-skade']
# 28
alleord['slag'] = ['slag-F-slag-i-slag', 'slag-4a', 'slag-4', 'slag-F-klokken-falder-i-slag', 'slag-6b', 'slag-1',
                   'slag-7', 'slag-2b', 'slag-F-på-slaget', 'slag-F-hjerte-springer-slag-over', 'slag-1c', 'slag-3',
                   'slag-1b', 'slag-6', 'slag-1a', 'slag-F-stor-i-slaget', 'slag-3a', 'slag-F-slag-i-bolledejen',
                   'slag-F-slag-på-tasken', 'slag-F-slag-i-luften', 'slag-F-slag-i-ansigtet', 'slag-6a',
                   'slag-F-slå-et-slag', 'slag-2c', 'slag-2', 'slag-2a', 'slag-F-frit-slag', 'slag-F-med-ét-slag']
# 11
alleord['stand'] = ['stand-F-få/bringe-i-stand', 'stand-F-holde-stand', 'stand-F-komme-i-stand',
                    'stand-F-gøre-(sætte,-lave,-..)-i-stand', 'stand-F-ude-af-stand-til', 'stand-4',
                    'stand-F-se-sig-i-stand-til', 'stand-1', 'stand-3', 'stand-2', 'stand-F-i-stand-til']
# 22
alleord['stykke'] = ['stykke-3a', 'stykke-2', 'stykke-1', 'stykke-5', 'stykke-F-slå-i-stykker', 'stykke-1a',
                     'stykke-F-en-4-5-(25,-tusind,-..)-stykker', 'stykke-F-i-stumper-og-stykker',
                     'stykke-F-når-det-kommer-til-stykket', 'stykke-5d', 'stykke-3',
                     'stykke-F-et-langt-stykke-(af-vejen)/et-langt-stykke-hen-ad-vejen', 'stykke-4c',
                     'stykke-F-i-stykker', 'stykke-5a', 'stykke-5b', 'stykke-2a', 'stykke-4', 'stykke-5c', 'stykke-4b',
                     'stykke-4a', 'stykke-4d']
# 12
alleord['top'] = ['top-1', 'top-2', 'top-F-toppen-af-isbjerget', 'top-2a', 'top-F-til-tops', 'top-2b',
                  'top-F-fra-top-til-tå', 'top-3', 'top-F-toppen-af/på-kransekagen', 'top-F-tånd/grå-i-toppen',
                  'top-F-på-toppen-af', 'top-1a']
# 10
alleord['vold'] = ['vold-F-gøre/øve-vold-mod/på', 'vold1-1', 'vold2-1', 'vold2-1a', 'vold1-1b', 'vold1-1a', 'vold1-1d',
                   'vold-F-med-vold-og-magt', 'vold1-2', 'vold1-1c']
# 10
alleord['kontakt'] = ['B-kontakt-2', 'B-kontakt-1a', 'B-kontakt-1d', 'B-kontakt-3', 'B-kontakt-3a', 'B-kontakt-1',
                      'B-kontakt-1b', 'B-kontakt-2a', 'B-kontakt-1c']  # 'B-kontakt-andet'
# 11
alleord['selskab'] = ['B-selskab-1b', 'B-selskab-1c', 'B-selskab-3', 'B-selskab-4a', 'B-selskab-2',
                      'B-selskab-F-holde-med-selskab', 'B-selskab-4', 'B-selskab-5', 'B-selskab-2a', 'B-selskab-1',
                      'B-selskab-1a']

# %%

# In[3]: Some synsets did not have examples in the DanNet data (12). I found the missing parts from the danish dictionary website ordnet.dk
no_examples = ['ansigt-1b', 'hul-4', 'kort-1', 'kort-1c', 'lys-1c', 'plade-6', 'slag-F-klokken-falder-i-slag',
               'slag-F-med-ét-slag', 'slag-F-på-slaget', 'stykke-2a', 'stykke-5', 'top-2b']

manually_found_examples = {'15549': ['månens', 'ansigt'],
                           '58363': ['naturlig', 'åbning', 'i', 'et', 'menneskes', 'krop'],
                           '20659': ['pengeinstitutterne', 'der', 'afgør', 'hvornår', 'en', 'kunde', 'skal', 'hæfte',
                                     'for', 'en', 'tyvs', 'misbrug', 'af', 'kort', 'og', ' kode'],
                           '58462': ['jeg', 'har', 'slukket', 'lyset', 'på', 'værelset', 'og', 'sidder', 'og',
                                     'skriver', 'i', 'stearinlysets', 'og', 'skærmens', 'forenede', 'skær'],
                           '18526': ['det', 'bliver', 'til', 'cirka', 'ti', 'kilowatt', 'det', 'svarer', 'sådan',
                                     'nogenlunde', 'til', 'et', 'komfur', 'med', 'fire', 'plader', 'og', 'en', 'ovn',
                                     'og', 'en', 'grill,', 'der', 'er', 'tændt', 'hele', 'tiden'],
                           '67203': ['ringenes', 'herre', 'ændrede', 'hendes', 'liv', 'med', 'ét', 'slag'],
                           '67204': ['jeg', 'ankom', 'til', 'hovedbanegården', 'på', 'slaget', '14'],
                           '35533': ['man', 'kan', 'operere', 'og', 'udskifte', 'et', 'stykke', 'af', 'en', 'blodåre'],
                           '28274': ['den', '21', 'år', 'gamle', 'teaterforening', 'kan', 'godt', 'lide',
                                     'traditionelle', 'stykker', 'med', 'sang'],
                           '63072': ['i', 'toppen', 'har', 'vi', 'ministeren', 'og', 'hans/hendes', 'stab', 'af',
                                     'magtfulde', 'departementschefer', 'og', 'direktører,', 'kontorchefer', 'og',
                                     'fuldmægtige']}

save_obj(key, 'semdax_to_DanNet')
save_obj(key, 'semdax_to_DanNet', save_json=True)
save_obj(alleord, 'word_to_semdax_label')
save_obj(alleord, 'word_to_semdax_label', save_json=True)
save_obj(manually_found_examples, 'manual_examples')
save_obj(manually_found_examples, 'manual_examples', save_json=True)