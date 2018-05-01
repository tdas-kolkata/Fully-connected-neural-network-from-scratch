import cv2
import numpy as np

########  controls  #########
n=3
layer=[64,400,40,70,n]
epoch=1000
alpha=0.01
initiation=False

##########################
#for character 0 and 1

no_of_weight_matrix=len(layer)-1
x=[]
y=[]

#process dataset
for i in range(n):
    for j in range(1,10):
        num=i+(j/10)
        img_name=str(num)+'.png'
        img=cv2.imread(img_name,0)
        a=np.asarray(img)
        a=a.reshape((64))
        k=np.repeat(-1,n)
        k[i]=1
        x.append(a)
        y.append(k)

x=np.array(x)
y=np.array(y)
print('training dataset is ready......')

#training data set is ready for supervised learning

def tan_h(x):
    return (np.tanh(x))
def d_tan_h(x):
    return(1-(x*x))

def check(pic,flag):
    pic_name=str(pic)+'.png'
    img=cv2.imread(pic_name,0)
    img=img.reshape((64))
    z_final=cal_output_test(img,weightlist)
    max_pos=0
    for i in range(1,len(z_final)):
        if z_final[i]>z_final[max_pos]:
            max_pos=i
    if flag==1:
        print(z_final)
    return max_pos

def total_check():
    right=wrong=0
    for i in range(n):
        for j in range(1,10):
            num=i+(j/10)
            print(num)
            res=check(num,0)
            if res==i:
                right=right+1
            else:
                wrong=wrong+1
            print(res)
        print('**************')
    print('right = '+str(right))
    print('wrong = '+str(wrong))

def commit(weightlist):
    weightlist=(np.array(weightlist))
    np.save('weightlist_for1&0.npy',weightlist)
    
def cal_output_test(x,weighlist):
    z=np.dot(x,weightlist[0])
    z=tan_h(z)
    for w in range(1,no_of_weight_matrix):
        z=np.dot(z,weightlist[w])
        z=tan_h(z)
    return z

def mean_sq_err(a):
        error=y-a
        sq_error=np.multiply(error,error)
        return (np.mean(sq_error))


def cal_output(x,weighlist):
    z=np.dot(x,weightlist[0])
    z=tan_h(z)
    node.append(z)
    for w in range(1,no_of_weight_matrix):
        z=np.dot(z,weightlist[w])
        z=tan_h(z)
        node.append(z)
    return z

def  find_delta_weight():
    i=len(node)-1
    error=y-node[i]
    error=2*(error*d_tan_h(node[i]))
    delta=np.dot(node[i-1].T,error)
    deltalist.append(delta)
    i=i-1
    while i >=0:
        w=weightlist[i+1]
        error=(np.dot(error,w.T))*d_tan_h(node[i])
        if i==0:
            delta=np.dot(x.T,error)
        else:
            delta=np.dot(node[i-1].T,error)
        deltalist.append(delta)
        i-=1
#strarting
        
weightlist=[]
deltalist=[]
node=[]

if initiation ==True:
    for i in range (len(layer)-1):
        w=np.random.randn(layer[i],layer[i+1])
        weightlist.append(w)
    print('weights initialised......')
else:
    weightlist=np.load('weightlist_for1&0.npy')


for num in range(epoch):
    a1=cal_output(x,weightlist)
    deltalist=[]
    find_delta_weight()
    node=[]
    rate=alpha*((num/epoch)*(num/epoch))
    for i in range(no_of_weight_matrix):
        weightlist[i]=weightlist[i]+alpha*deltalist[-(i+1)]
    deltalist=[]
    if num%100==0:
        print(mean_sq_err(a1))
        commit(weightlist)
    deltalist=[]

total_check()









