import pickle

from cldm.model import create_model, load_state_dict

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'))

with open('./models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
