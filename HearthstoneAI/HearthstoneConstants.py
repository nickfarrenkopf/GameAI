
### CONSTANTS ###

networks = dict()
json_data = dict()

elements = ['heroS', 'heroO', 'heroPowerS', 'heroPowerO', 'manaS', 'manaO',
            'weaponS', 'weaponO', 'handS', 'handO', 'boardS', 'boardO',
            'deckS', 'deckO']


### HERO ###

heroS = dict()
heroS['center'] = [820, 965]
heroS['image_size'] = [192, 192]
heroS['network_name'] = 'hero'

heroO = dict()
heroO['center'] = [195, 965]
heroO['image_size'] = [192, 192]
heroO['network_name'] = 'hero'

json_data['heroS'] = heroS
json_data['heroO'] = heroO

networks['hero'] = dict()
networks['hero']['n_latent'] = 16
networks['hero']['network_size'] = [64, 64]
networks['hero']['every_n'] = 8


### HERO POWER ###

heroPowerS = dict()
heroPowerS['center'] = [1135, 823]
heroPowerS['image_size'] = [160, 160]
heroPowerS['network_name'] = 'heroPower'

heroPowerO = dict()
heroPowerO['center'] = [1135, 235]
heroPowerO['image_size'] = [160, 160]
heroPowerO['network_name'] = 'heroPower'

json_data['heroPowerS'] = heroPowerS
json_data['heroPowerO'] = heroPowerO

networks['heroPower'] = dict()
networks['heroPower']['n_latent'] = 8
networks['heroPower']['network_size'] = [64, 64]
networks['heroPower']['every_n'] = 16


### MANA ###

manaS = dict()
manaS['center'] = [1000, 1262]
manaS['image_size'] = [32, 64]
manaS['network_name'] = 'mana'

manaO = dict()
manaO['center'] = [67, 1229]
manaO['image_size'] = [32, 64]
manaO['network_name'] = 'mana'

json_data['manaS'] = manaS
json_data['manaO'] = manaO

networks['mana'] = dict()
networks['mana']['n_latent'] = 2
networks['mana']['network_size'] = [16, 32]
networks['mana']['every_n'] = 16


### WEAPON ###

weaponS = dict()
weaponS['center'] = [765, 830]
weaponS['image_size'] = [160, 160]
weaponS['network_name'] = 'weapon'

weaponO = dict()
weaponO['center'] = [785, 238]
weaponO['image_size'] = [160, 160]
weaponO['network_name'] = 'weapon'

json_data['weaponS'] = weaponS
json_data['weaponO'] = weaponO

networks['weapon'] = dict()
networks['weapon']['n_latent'] = 4
networks['weapon']['network_size'] = [64, 64]
networks['weapon']['every_n'] = 16


### HAND ###

handS = dict()
handS['center'] = [924, 1000]
handS['image_size'] = [680, 160]
handS['network_name'] = 'handS'

handO = dict()
handO['center'] = [914, 48]
handO['image_size'] = [366, 76]
handO['network_name'] = 'handO'

json_data['handS'] = handS
json_data['handO'] = handO

networks['handS'] = dict()
networks['handS']['n_latent'] = 32
networks['handS']['network_size'] = [64, 256]
networks['handS']['every_n'] = 4

networks['handO'] = dict()
networks['handO']['n_latent'] = 16
networks['handO']['network_size'] = [32, 128]
networks['handO']['every_n'] = 8


### DECK ###

deckS = dict()
deckS['center'] = [1805, 440]
deckS['image_size'] = [220, 840]
deckS['network_name'] = 'deck'

deckO = dict()
deckO['center'] = [120, 520]
deckO['image_size'] = [220, 770]
deckO['network_name'] = 'deck'

json_data['deckS'] = deckS
json_data['deckO'] = deckO

networks['deck'] = dict()
networks['deck']['n_latent'] = 32
networks['deck']['network_size'] = [256, 64]
networks['deck']['every_n'] = 64


### BOARD ###

boardS = dict()
boardS['center'] = [960, 593]
boardS['image_size'] = [980, 160]
boardS['network_name'] = 'board'

boardO = dict()
boardO['center'] = [960, 410]
boardO['image_size'] = [980, 160]
boardO['network_name'] = 'board'

json_data['boardS'] = boardS
json_data['boardO'] = boardO

networks['board'] = dict()
networks['board']['n_latent'] = 32
networks['board']['network_size'] = [64, 256]
networks['board']['every_n'] = 2


