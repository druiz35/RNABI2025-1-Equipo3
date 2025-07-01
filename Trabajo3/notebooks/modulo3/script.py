from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model

# Inputs
user_input = Input(shape=(1,), name='user_id')
place_input = Input(shape=(1,), name='place_id')
grupo_input = Input(shape=(1,), name='grupo_id')
categoria_input = Input(shape=(1,), name='categoria_id')
num_features_input = Input(shape=(2,), name='numerical_features')  # e.g., presupuesto, distancia

# Embeddings
embedding_dim = 32

user_emb = Embedding(input_dim=1000, output_dim=embedding_dim, name='user_embedding')(user_input)
place_emb = Embedding(input_dim=5000, output_dim=embedding_dim, name='place_embedding')(place_input)
grupo_emb = Embedding(input_dim=4, output_dim=embedding_dim//2, name='grupo_embedding')(grupo_input)
categoria_emb = Embedding(input_dim=10, output_dim=embedding_dim//2, name='categoria_embedding')(categoria_input)

# Flatten embeddings
user_vec = Flatten()(user_emb)
place_vec = Flatten()(place_emb)
grupo_vec = Flatten()(grupo_emb)
categoria_vec = Flatten()(categoria_emb)

# Concatenate all inputs
concat = Concatenate()([user_vec, place_vec, grupo_vec, categoria_vec, num_features_input])

# MLP
x = Dense(128, activation='relu')(concat)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Model
model = Model(inputs=[user_input, place_input, grupo_input, categoria_input, num_features_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

model.summary()


# Suponiendo que tienes arrays NumPy o pandas:
# user_ids, place_ids, grupo_ids, categoria_ids, features_numericas, labels

model.fit(
    x=[user_ids, place_ids, grupo_ids, categoria_ids, features_numericas],
    y=labels,
    batch_size=64,
    epochs=10,
    validation_split=0.2
)

