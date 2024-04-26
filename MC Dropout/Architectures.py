#%%
import torch
import torch.nn as nn



#%%
class Network_16(nn.Module) :
    def __init__(self, hidden_size, p) :
        super(Network_16,self).__init__()

        self.name = 'Network_16'

        self.Layer1 = nn.Sequential(
            nn.Linear(
            in_features=1,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )
        
        self.dropout1 = nn.Dropout(p)

        self.Layer2 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout2 = nn.Dropout(p)

        self.Layer3 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout3 = nn.Dropout(p)

        self.Layer4 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout4 = nn.Dropout(p)

        self.Layer5 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout5 = nn.Dropout(p)

        self.Layer6 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout6 = nn.Dropout(p)

        self.Layer7 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout7 = nn.Dropout(p)

        self.Layer8 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout8 = nn.Dropout(p)


        self.Layer9 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout9 = nn.Dropout(p)


        self.Layer10 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout10 = nn.Dropout(p)


        self.Layer11 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout11 = nn.Dropout(p)


        self.Layer12 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout12 = nn.Dropout(p)

        self.Layer13 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout13 = nn.Dropout(p)

        self.Layer14 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout14 = nn.Dropout(p)


        self.Layer15 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout15 = nn.Dropout(p)


        self.Layer16 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout16 = nn.Dropout(p)

        self.output = nn.Linear(
                        in_features=hidden_size,
                        out_features = 1,
                        bias=True)
        

    def forward(self,x) :
        x = self.Layer1(x)
        x = self.dropout1(x)
        x = self.Layer2(x)
        x = self.dropout2(x)
        x = self.Layer3(x)
        x = self.dropout3(x)
        x = self.Layer4(x)
        x = self.dropout4(x)
        x = self.Layer5(x)
        x = self.dropout5(x)
        x = self.Layer6(x)
        x = self.dropout6(x)
        x = self.Layer7(x)
        x = self.dropout7(x)
        x = self.Layer8(x)
        x = self.dropout8(x)
        x = self.Layer9(x)
        x = self.dropout9(x)
        x = self.Layer10(x)
        x = self.dropout10(x)
        x = self.Layer11(x)
        x = self.dropout11(x)
        x = self.Layer12(x)
        x = self.dropout12(x)
        x = self.Layer13(x)
        x = self.dropout13(x)
        x = self.Layer14(x)
        x = self.dropout14(x)
        x = self.Layer15(x)
        x = self.dropout15(x)
        x = self.Layer16(x)
        x = self.dropout16(x)
        out = self.output(x)
        return out

# %%
class Network_8(nn.Module) :
    def __init__(self, hidden_size, p) :
        super(Network_8,self).__init__()

        self.name = 'Network_8'

        self.Layer1 = nn.Sequential(
            nn.Linear(
            in_features=1,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )
        
        self.dropout1 = nn.Dropout(p)

        self.Layer2 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout2 = nn.Dropout(p)

        self.Layer3 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout3 = nn.Dropout(p)

        self.Layer4 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout4 = nn.Dropout(p)

        self.Layer5 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout5 = nn.Dropout(p)

        self.Layer6 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout6 = nn.Dropout(p)

        self.Layer7 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout7 = nn.Dropout(p)

        self.Layer8 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout8 = nn.Dropout(p)



        self.output = nn.Linear(
                        in_features=hidden_size,
                        out_features = 1,
                        bias=True)
        

    def forward(self,x) :
        x = self.Layer1(x)
        x = self.dropout1(x)
        x = self.Layer2(x)
        x = self.dropout2(x)
        x = self.Layer3(x)
        x = self.dropout3(x)
        x = self.Layer4(x)
        x = self.dropout4(x)
        x = self.Layer5(x)
        x = self.dropout5(x)
        x = self.Layer6(x)
        x = self.dropout6(x)
        x = self.Layer7(x)
        x = self.dropout7(x)
        x = self.Layer8(x)
        x = self.dropout8(x)
        out = self.output(x)
        return out
    


# %%
class Network_4(nn.Module) :
    def __init__(self, hidden_size, p) :
        super(Network_4,self).__init__()

        self.name = 'Network_4'

        self.Layer1 = nn.Sequential(
            nn.Linear(
            in_features=1,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )
        
        self.dropout1 = nn.Dropout(p)

        self.Layer2 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout2 = nn.Dropout(p)

        self.Layer3 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout3 = nn.Dropout(p)

        self.Layer4 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout4 = nn.Dropout(p)


        self.output = nn.Linear(
                        in_features=hidden_size,
                        out_features = 1,
                        bias=True)
        

    def forward(self,x) :
        x = self.Layer1(x)
        x = self.dropout1(x)
        x = self.Layer2(x)
        x = self.dropout2(x)
        x = self.Layer3(x)
        x = self.dropout3(x)
        x = self.Layer4(x)
        x = self.dropout4(x)
        out = self.output(x)
        return out




# %%
class Network_2(nn.Module) :
    def __init__(self, hidden_size, p) :
        super(Network_2,self).__init__()

        self.name = 'Network_2'

        self.Layer1 = nn.Sequential(
            nn.Linear(
            in_features=1,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )
        
        self.dropout1 = nn.Dropout(p)

        self.Layer2 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout2 = nn.Dropout(p)


        self.output = nn.Linear(
                        in_features=hidden_size,
                        out_features = 1,
                        bias=True)
        

    def forward(self,x) :
        x = self.Layer1(x)
        x = self.dropout1(x)
        x = self.Layer2(x)
        x = self.dropout2(x)
        out = self.output(x)
        return out




# %%
