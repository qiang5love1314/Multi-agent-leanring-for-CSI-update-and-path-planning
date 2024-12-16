import numpy as np
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
np.random.seed(123)

def build_generator(latent_dim, label_dim, data_dim):
    generator_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(label_dim,))
    x = Concatenate()([generator_input, label_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(data_dim, activation='linear')(x)
    generator = Model([generator_input, label_input], x)
    return generator

def build_discriminator(data_dim, label_dim):
    discriminator_input = Input(shape=(data_dim,))
    label_input = Input(shape=(label_dim,))
    x = Concatenate()([discriminator_input, label_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model([discriminator_input, label_input], x)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['accuracy'])
    return discriminator

def build_gan(generator, discriminator):
    discriminator.trainable = False
    generator_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(label_dim,))
    x = generator([generator_input, label_input])
    gan_output = discriminator([x, label_input])
    gan = Model([generator_input, label_input], gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002))
    return gan

def train_gan(generator, discriminator, gan, gauss_inputs, gauss_labels, epochs=1000, batch_size=32):
    generator.compile(optimizer='adam', loss='mse')
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    best_accuracy = 0

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = gauss_labels[np.random.randint(0, gauss_labels.shape[0], batch_size)]
        generated_data = generator.predict([noise, sampled_labels])
        
        real_data_batch = gauss_inputs[np.random.randint(0, gauss_inputs.shape[0], batch_size)]
        fake_labels = np.zeros((batch_size, 1))
        real_labels = np.ones((batch_size, 1))
        
        d_loss_fake = discriminator.train_on_batch([generated_data, sampled_labels], fake_labels)
        d_loss_real = discriminator.train_on_batch([real_data_batch, sampled_labels], real_labels)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch([noise, sampled_labels], valid_labels)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, Model Accuracy: {d_loss[1] * 100}%, G Loss: {g_loss}")
        if d_loss[1] > best_accuracy:
            best_accuracy = d_loss[1]

    return best_accuracy

# Set parameters
latent_dim = 4500
label_dim = 2

# # Build and compile the models
# data_dim = 4500  # Replace with actual data dimension
# generator = build_generator(latent_dim, label_dim, data_dim)
# discriminator = build_discriminator(data_dim, label_dim)
# gan = build_gan(generator, discriminator)

# # Dummy data for example purposes (replace with actual data)
# gauss_inputs = np.random.normal(0, 1, (100, data_dim))
# gauss_labels = np.random.normal(0, 1, (100, label_dim))

# # Train the GAN
# train_gan(generator, discriminator, gan, gauss_inputs, gauss_labels, epochs=100, batch_size=32)