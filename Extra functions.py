
def visualize_box_plots(losses_list, title):
    
    fig, ax = plt.subplots()
    box_data = []
    labels = []
    
    for i, losses in enumerate(losses_list):
        box_data.append(losses)
        label = f"Model {i + 1}"
        labels.append(label)
    
    ax.boxplot(box_data, patch_artist=True, labels=labels)
    ax.set_ylabel('Loss')
    ax.set_title(title)
    
    plt.show()

def plot_results(results):
    
    methods = sorted(set([(grayscale, normalize, blur) for grayscale, normalize, blur, _, _ in results]))
    losses = [np.mean([loss for grayscale, normalize, blur, loss, _ in results if (grayscale, normalize, blur) == method]) for method in methods]
    times = [np.mean([time for grayscale, normalize, blur, _, time in results if (grayscale, normalize, blur) == method]) for method in methods]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    axs[0].bar(x - width/2, losses, width, label='Mean Loss')
    axs[0].set_title('Mean Loss for Preprocessing Methods', fontsize=14)
    axs[0].set_xticks(x)
    method_labels = [f"Grayscale: {method[0]}\nNormalize: {method[1]}\nBlur: {method[2]}" for method in methods]
    axs[0].set_xticklabels(method_labels, rotation=45, ha='right', fontsize=10)
    axs[0].legend()
    
    for i, loss_value in enumerate(losses):
        axs[0].annotate(f"{loss_value:.4f}", xy=(x[i] - width/4, loss_value + 0.01), fontsize=10)
    
    axs[1].bar(x - width/2, times, width, label='Mean Time')
    axs[1].set_title('Mean Time for Preprocessing Methods', fontsize=14)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(method_labels, rotation=45, ha='right', fontsize=10)
    axs[1].legend()
    
    for i, time_value in enumerate(times):
        axs[1].annotate(f"{time_value:.4f}", xy=(x[i] - width/4, time_value + 0.01), fontsize=10)
    
    plt.tight_layout()
    plt.show()


def apply_preprocessing(images, pts, img_size, grayscale, normalize, blur):
    processed_images = []
    resized_pts = []
    
    for img, kp in zip(images, pts):
        resized_img = cv.resize(img, img_size)
        
        if grayscale:
            resized_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
        
        if normalize:
            resized_img = resized_img / 255.0
        
        if blur:
            resized_img = cv.GaussianBlur(resized_img, (5, 5), 0)
        
        processed_images.append(resized_img)
        
        h, w = img.shape[:2]
        new_h, new_w = img_size
        resized_kp = kp * [new_w / w, new_h / h]
        resized_pts.append(resized_kp)
    
    processed_images = np.array(processed_images)
    points = np.array(resized_pts)
    
    return processed_images, points


def compare_preprocessing_methods(images, pts, img_size=(128, 128)):
    
    results = []
    methods = product([True, False], repeat=3)
    
    for grayscale, normalize, blur in methods:
        start_time = time.time()
        preprocessed_images, points = apply_preprocessing(images, pts, img_size, grayscale, normalize, blur)
        
        X_train_images, X_test_images, y_train_labels, y_test_labels = train_test_split(preprocessed_images, points, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        X_train_images_flat = X_train_images.reshape(X_train_images.shape[0], -1)
        X_test_images_flat = X_test_images.reshape(X_test_images.shape[0], -1)
        
        X_train_images_pca = pca.fit_transform(X_train_images_flat)
        X_test_images_pca = pca.transform(X_test_images_flat)
        
        linear_reg_model, loss = linearRegression(X_train_images_pca, X_test_images_pca, y_train_labels, y_test_labels)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results.append((grayscale, normalize, blur, loss, execution_time))
    
    plot_results(results)


def augment_data(images, pts):
    
    num_images = len(images)
    flipped_images = [cv.flip(img, 1) for img in images]
    
    # Flip the points horizontally
    flipped_pts = []
    for img, kp in zip(images, pts):
        h, w = img.shape[:2]
        flipped_kp = np.array([(w - 1 - x, y) for x, y in kp])
        flipped_pts.append(flipped_kp)
    
    # Combine original and flipped data
    augmented_images = np.concatenate((images, flipped_images), axis=0)
    augmented_pts = np.concatenate((pts, flipped_pts), axis=0)
    
    # Visualize original and augmented data
    if SHOW_AUGMENT_DATA:
        for i in range(num_images):
            plt.figure(figsize=(8, 4))
            
            # Original image with points
            plt.subplot(1, 2, 1)
            plt.imshow(images[i])
            plt.scatter(pts[i][:, 0], pts[i][:, 1], color='red', s=10)
            plt.title(f'Image {i+1}')
            
            # Flipped image with points
            plt.subplot(1, 2, 2)
            plt.imshow(flipped_images[i])
            plt.scatter(flipped_pts[i][:, 0], flipped_pts[i][:, 1], color='red', s=10)
            plt.title(f'Image {num_images + i + 1} (Flipped)')
            
            plt.show()
            
    return augmented_images, augmented_pts







