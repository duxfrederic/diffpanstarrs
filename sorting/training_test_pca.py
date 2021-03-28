import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

###############################################################################################

def imshow(img):
	print(img)
	print(img.shape)
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

###############################################################################################
torch.manual_seed(7)

transform = transforms.ToTensor()

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
y_train = y_train.astype(np.float32)
x_train[np.isnan(x_train)] = 0

good = np.where(y_train==1)[0]
bad = np.where(y_train==0)[0]
np.random.shuffle(bad)

where = np.append(bad[:len(good)], good)
np.random.shuffle(where)
where = (where,)
x_train, y_train = x_train[where], y_train[where]

m, s = x_train.mean(), x_train.std()
mm = m + 3*s
mn = m - 3*s
x_train[x_train>mm] = mm
x_train[x_train<mn] = mn
x_train /= mm
x_train -= x_train.mean()

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
#%%

# trainset = dset.MNIST(root='Data/', train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

# testset = dset.MNIST(root='Data/', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

N_train = len(x_train)
# N_test  = len(testset)
# x = torch.stack([trainset[i][0] for i in range(N_train)])  
# labels = torch.stack([trainset[i][1] for i in range(N_train)]) 

# x_test = torch.stack([testset[i][0] for i in range(N_test)])  
# labels_test = torch.stack([testset[i][1] for i in range(N_test)]) 


mu = x_train.mean([-2,-1])  
# mu_test = x_test.mean(0)
# x_test = x_test - mu
# x_test = x_test.view(x_test.size(0), -1) 

x_train = x_train - mu[:, np.newaxis, np.newaxis]
# x_train = x_train.view(x_train.size(0), -1) 

cov = x_train.t() @ x_train / float(N_train)
eig, eigv = cov.symeig(eigenvectors=True)

criterion = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4


accu = {}
path = "Model/"

for s in [10]:
	factor = np.sqrt(eig[-s:])
	proj = x_train @ (eigv[:,-s:]/factor)
	
	print (proj.size())	

# 	proj_test = x_test @ (eigv[:,-s:]/factor)

	model = torch.nn.Sequential( torch.nn.Linear(s, 800), torch.nn.ReLU(), torch.nn.Linear(800, 10) )
	
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	
	running_loss = 0.0
	
	losses = []
	N = 1000
	
	train = 1
	if train:
		ts = time.time()
		for kk in range(N_train):
			image = proj[kk,:].view(1, s)
			#print ("image", image.shape, image)
			label = y_train[kk].view(1)
			# ("label", label.shape, label)
			#image = image.view(1, -1)
			
			optimizer.zero_grad()
			out = model( image )

			#print (out.shape)
	
			loss = criterion(out, label)
	
			loss.backward()
	
			optimizer.step()
		
			running_loss += loss.item()
			if kk%N == 0:
				print (kk)
				print(running_loss/N)
				losses.append( running_loss/float(N) )
				running_loss = 0.0
	
			#if i==1000:
				#break
	
		losses = np.array( losses )
	
		#np.save("Model/losses_" + str(s), losses)
	
		#torch.save(model.state_dict(), "Model/nn_" + str(s) + ".dat")
	
		print ("Train, time elapsed: {0:.6f}".format(time.time() - ts))
	else:
		losses = np.load("Model/losses1.npy")

	numer = {}
	numer_correct = {}

	for i in range(10):
		numer[i] = 0.0
		numer_correct[i] = 0.0

# 	test = 1
# 	if test:
# 		correct = 0
# 		total = 0
# 		with torch.no_grad():
# 			for kk in range(N_test):
# 				images = proj_test[kk,:].view(1, s)
# 				labelsss = labels_test[kk].view(1)
# 				outputs = model(images)
# 				_, predicted = torch.max(outputs.data, 1)

# 				k = labelsss.item()
# 				numer[k] += 1.0
# 				total += labelsss.size(0)
# 				correct += (predicted == labelsss).sum().item()
# 				numer_correct[k] +=  (predicted == labelsss).sum().item()
# 	print("Accuracy of the network on the 10000 test images: {0:.4%}".format(correct / total))

# 	resultat = {}
# 	for k in numer.keys():
# 		resultat[k] =  numer_correct[k]/numer[k]
# 		print (k, numer_correct[k]/numer[k])
# 	resultat["total"] = (correct / total)
	
# 	res = 0.0
# 	res_c = 0.0
# 	for k in numer.keys():
# 		res += numer[k]
# 		res_c += numer_correct[k]

# 	print (res_c/res)
# 		
# 	accu[s] = resultat
np.save(path + "accuracy_pca.npy", accu)
#"""