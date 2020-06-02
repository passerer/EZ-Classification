def split_rand(file:np.ndarray,label:np.ndarray,p=0.2,seed=None):
    size = len(file)
    assert size == len(label),"file list len is not equal to label list len"
    if seed is not None: np.random.seed(seed)
    print("split train and val file...")
    per = np.random.permutation(size)
    label = label[per,...]
    file = file[per,...]
    val_file = file[0:int(p*size),...]
    val_label = label[0:int(p*size),...]
    
    train_file = file[int(p*size):size,...]
    train_label = label[int(p*size):size,...]
    return train_file,train_label,val_file,val_label

def show_shape_range(img_file):
    width = []
    height = []
    print("collecting shape information...")
    #for i in tqdm(range(len(img_file))):
    for i in range(len(img_file)):
        img = cv2.imread(img_file[i])
        width.append(img.shape[1])
        height.append(img.shape[0])
    plt.xlabel("width")
    plt.ylabel("height")
    plt.scatter(width,height,s=1.)
    plt.show()
    
def show_batch(dataset,n=16):
    if n>25:
        print("can't plot so many images")
        n = 25
    row = int(math.sqrt(n))
    col = n // row
    n = row*col
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    fig=plt.figure(figsize=(row*20,col*20))
    for i in range(1,n+1):
        img,label = dataset[i-1]
        img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        img *= std
        img += mean
        img = np.clip(img,0.,1.)
    #    img = cv2.resize(img, dsize=(20,20))
        ax = fig.add_subplot(row, col, i)
    #    title = "\n".join(wrap(label.replace('_', ' '), 25))
    #    ax.set_title(title, fontsize=18, fontweight='bold', x=0.5, y=0.001, bbox=dict(facecolor='white', alpha=0.5))
        plt.axis('off') 
        plt.imshow(img)
    plt.show()