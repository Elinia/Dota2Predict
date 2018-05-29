import tkinter as tk
import numpy as np
import mxnet as mx

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.mx_model = self.load_mx_model()
        self.heroes_text = self.get_heroes()
        self.heroes_text_tuple = tuple(self.heroes_text)
        self.v1 = tk.StringVar()
        self.v1.set(self.heroes_text[0])
        self.v2 = tk.StringVar()
        self.v2.set(self.heroes_text[0])
        self.v3 = tk.StringVar()
        self.v3.set(self.heroes_text[0])
        self.v4 = tk.StringVar()
        self.v4.set(self.heroes_text[0])
        self.v5 = tk.StringVar()
        self.v5.set(self.heroes_text[0])
        self.v6 = tk.StringVar()
        self.v6.set(self.heroes_text[0])
        self.v7 = tk.StringVar()
        self.v7.set(self.heroes_text[0])
        self.v8 = tk.StringVar()
        self.v8.set(self.heroes_text[0])
        self.v9 = tk.StringVar()
        self.v9.set(self.heroes_text[0])
        self.v10 = tk.StringVar()
        self.v10.set(self.heroes_text[0])
        self.predict_text = tk.StringVar()
        self.predict_text.set('')
        self.suggest1 = tk.StringVar()
        self.suggest1.set('')
        self.suggest2 = tk.StringVar()
        self.suggest2.set('')
        self.suggest3 = tk.StringVar()
        self.suggest3.set('')
        self.suggest4 = tk.StringVar()
        self.suggest4.set('')
        self.suggest5 = tk.StringVar()
        self.suggest5.set('')
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text='Ally heroes').grid(row=0, column=0)
        tk.Label(self, text='Enemy heroes').grid(row=0, column=1)
        tk.OptionMenu(self, self.v1, *self.heroes_text_tuple).grid(row=1, column=0, sticky='w')
        tk.OptionMenu(self, self.v2, *self.heroes_text_tuple).grid(row=2, column=0, sticky='w')
        tk.OptionMenu(self, self.v3, *self.heroes_text_tuple).grid(row=3, column=0, sticky='w')
        tk.OptionMenu(self, self.v4, *self.heroes_text_tuple).grid(row=4, column=0, sticky='w')
        tk.OptionMenu(self, self.v5, *self.heroes_text_tuple).grid(row=5, column=0, sticky='w')
        tk.OptionMenu(self, self.v6, *self.heroes_text_tuple).grid(row=1, column=1, sticky='e')
        tk.OptionMenu(self, self.v7, *self.heroes_text_tuple).grid(row=2, column=1, sticky='e')
        tk.OptionMenu(self, self.v8, *self.heroes_text_tuple).grid(row=3, column=1, sticky='e')
        tk.OptionMenu(self, self.v9, *self.heroes_text_tuple).grid(row=4, column=1, sticky='e')
        tk.OptionMenu(self, self.v10, *self.heroes_text_tuple).grid(row=5, column=1, sticky='e')

        tk.Button(self, text="Predict", fg="red", command=self.predict).grid(row=6, column=0)
        tk.Button(self, text="Suggest", fg="red", command=self.suggest).grid(row=0, column=2)

        tk.Label(self, textvariable=self.predict_text).grid(row=6, column=1)

        tk.Label(self, textvariable=self.suggest1).grid(row=1, column=2)
        tk.Label(self, textvariable=self.suggest2).grid(row=2, column=2)
        tk.Label(self, textvariable=self.suggest3).grid(row=3, column=2)
        tk.Label(self, textvariable=self.suggest4).grid(row=4, column=2)
        tk.Label(self, textvariable=self.suggest5).grid(row=5, column=2)

    def to4d(self, img):
        return img.reshape(img.shape[0], 1, 1, 114)

    def load_mx_model(self):
        sym, arg_params, aux_params = mx.model.load_checkpoint('mx_mlp', 10)
        mx_model = mx.mod.Module(symbol=sym)
        struct = np.array([[0 for i in range(114)]])
        iter = mx.io.NDArrayIter(self.to4d(struct), np.array([1]))
        mx_model.bind(data_shapes=iter.provide_data, label_shapes=iter.provide_label)
        mx_model.set_params(arg_params, aux_params)
        return mx_model

    def predict(self):
        self.xT = [0 for i in range(114)]
        self.set_hero(self.v1.get(), 1)
        self.set_hero(self.v2.get(), 1)
        self.set_hero(self.v3.get(), 1)
        self.set_hero(self.v4.get(), 1)
        self.set_hero(self.v5.get(), 1)
        self.set_hero(self.v6.get(), -1)
        self.set_hero(self.v7.get(), -1)
        self.set_hero(self.v8.get(), -1)
        self.set_hero(self.v9.get(), -1)
        self.set_hero(self.v10.get(), -1)
        res = self.mx_model.predict(mx.io.NDArrayIter(self.to4d(np.array([self.xT])), np.array([1])))
        self.predict_text.set('Win rate: ' + str(res[0][1].asnumpy()[0]))

    def suggest(self):
        self.xT = np.array([0 for i in range(114)])
        self.set_hero(self.v1.get(), 1)
        self.set_hero(self.v2.get(), 1)
        self.set_hero(self.v3.get(), 1)
        self.set_hero(self.v4.get(), 1)
        self.set_hero(self.v5.get(), 1)
        self.set_hero(self.v6.get(), -1)
        self.set_hero(self.v7.get(), -1)
        self.set_hero(self.v8.get(), -1)
        self.set_hero(self.v9.get(), -1)
        self.set_hero(self.v10.get(), -1)

        heroes_arr = []
        hero_arr = []
        for hero_text in self.heroes_text:
            if self.hero_exists(hero_text):
                continue
            self.set_hero(hero_text, 1)
            heroes_arr.append(self.xT.copy())
            self.set_hero(hero_text, 0)
            hero_arr.append(hero_text)
        res = self.mx_model.predict(mx.io.NDArrayIter(self.to4d(np.array(heroes_arr))))
        win_rate = []
        for i in range(len(hero_arr)):
            win_rate.append({'hero': hero_arr[i], 'win_rate': res[i][1].asnumpy()[0]})
        win_rate_sorted = sorted(win_rate, key=lambda x: (x['win_rate']), reverse=True)
        self.suggest1.set(win_rate_sorted[0]['hero'] + ' with win rate: ' + str(win_rate_sorted[0]['win_rate']))
        self.suggest2.set(win_rate_sorted[1]['hero'] + ' with win rate: ' + str(win_rate_sorted[1]['win_rate']))
        self.suggest3.set(win_rate_sorted[2]['hero'] + ' with win rate: ' + str(win_rate_sorted[2]['win_rate']))
        self.suggest4.set(win_rate_sorted[3]['hero'] + ' with win rate: ' + str(win_rate_sorted[3]['win_rate']))
        self.suggest5.set(win_rate_sorted[4]['hero'] + ' with win rate: ' + str(win_rate_sorted[4]['win_rate']))

    def get_heroes(self):
        heroes_text = ['0,未选择,None']
        with open('heroid_114.txt', 'r', encoding='utf8') as f:
            for line in f.readlines():
                heroes_text.append(line.strip())
        return heroes_text

    def set_hero(self, hero_text, set_to):
        hero_id = int(hero_text.split(',')[0])
        if hero_id is not 0:
            self.xT[hero_id-1] = set_to

    def hero_exists(self, hero_text):
        hero_id = int(hero_text.split(',')[0])
        if self.xT[hero_id-1] != 0:
            return True
        else:
            return False


root = tk.Tk()
app = Application(master=root)
app.mainloop()
