import numpy as np
import matplotlib.pyplot as plt
import os

import pennylane as qml

# dossier de sauvegarde
dossier_sortie = "resultats_classifieur"
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)
    print(f"Dossier '{dossier_sortie}' créé pour les sauvegardes.")
np.random.seed(0)

# device 2 qubits
dev = qml.device("default.qubit", wires=2)


# génération de données
def generer_donnees(nb_points=200):
    X = []
    y = []

    for i in range(nb_points):
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        label = 1 if x1**2 + x2**2 < 0.5 else -1
        X.append([x1, x2])
        y.append(label)

    return np.array(X), np.array(y)


# circuit avec 2 qubits
@qml.qnode(dev)
def circuit(x, params):

    # encodage simple des données
    qml.RY(x[0] * np.pi, wires=0)
    qml.RY(x[1] * np.pi, wires=1)

    # paramètres du modèle
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.RZ(params[3], wires=1)

    # on crée un lien entre les 2 qubits
    qml.CNOT(wires=[0, 1])

    # encore un peu de flexibilité
    qml.RY(params[4], wires=0)
    qml.RY(params[5], wires=1)

    return qml.expval(qml.PauliZ(0))

# dessin du circuit
print("Dessin du circuit (mode console) :")
print(qml.draw(circuit)([0.5, -0.3], [0.1, 0.7, 0.2, -0.4, 0.3, -0.2]))
print("Dessin du circuit amélioré :")

qml.drawer.use_style("default")
fig_circuit, ax = qml.draw_mpl(circuit)([0.5, -0.3], [0.1, 0.7, 0.2, -0.4, 0.3, -0.2])
fig_circuit.suptitle("Circuit variationnel 2 qubits (6 paramètres)", fontsize=12)
plt.tight_layout()

chemin_circuit = os.path.join(dossier_sortie, "circuit.png")
fig_circuit.savefig(chemin_circuit, dpi=150, bbox_inches="tight")
print(f"Circuit sauvegardé : {chemin_circuit}")
plt.show()


# prédiction
def predire(x, params):
    val = circuit(x, params)
    return 1 if val >= 0 else -1


# coût MSE
def cout(X, y, params):
    erreur = 0

    for i in range(len(X)):
        pred = circuit(X[i], params)
        erreur += (pred - y[i])**2

    return erreur / len(X)

# entraînement
def entrainer(X, y, iterations=100, lr=0.2):

    meilleur_params = None
    meilleur_score = float('inf')

    for essai in range(1):

        params = np.random.uniform(-1, 1, 6)

        for i in range(iterations):

            grad = np.zeros_like(params)
            eps = 1e-3

            indices = np.random.choice(len(X), size=40, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            cout_actuel = cout(X_batch, y_batch, params)

            for j in range(len(params)):
                params_tmp = params.copy()
                params_tmp[j] += eps

                grad[j] = (cout(X_batch, y_batch, params_tmp) - cout_actuel) / eps

            params = params - lr * grad

            # on réduit un peu le pas vers la fin
            if i == 50:
                lr = lr * 0.5

            if i % 50 == 0 and essai == 0:
                print("it", i, "cout =", cout_actuel)

        c_final = cout(X, y, params)

        if c_final < meilleur_score:
            meilleur_score = c_final
            meilleur_params = params.copy()

        print("essai", essai+1, "cout final =", round(c_final, 1))

    return meilleur_params


# précision
def calculer_accuracy(X, y, params):
    correct = 0
    for i in range(len(X)):
        if predire(X[i], params) == y[i]:
            correct += 1
    return correct / len(X)


print("\n Préparation des données")

X_all, y_all = generer_donnees(300)
indices = np.random.permutation(len(X_all))
X_all, y_all = X_all[indices], y_all[indices]

X_train, y_train = X_all[:200], y_all[:200]
X_test, y_test = X_all[200:], y_all[200:]

print(f"Entraînement : {len(X_train)} points")
print(f"Test         : {len(X_test)} points")

print("\n" + "=" * 50)
print("Entraînement (plusieurs essais, meilleur gardé)")

params = entrainer(X_train, y_train)

acc_train = calculer_accuracy(X_train, y_train, params)
acc_test = calculer_accuracy(X_test, y_test, params)

print("\n" + "=" * 50)
print("Résultats")
print(f"Accuracy entraînement : {acc_train:.2%}")
print(f"Accuracy test         : {acc_test:.2%}")
print(f"Paramètres : {np.round(params, 4)}")


# graphique
print("\nGénération du graphique...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# entraînement
for i in range(len(X_train)):
    if y_train[i] == 1:
        ax1.scatter(X_train[i][0], X_train[i][1], color='blue', alpha=0.6,
                    label='intérieur (+1)' if i == 0 else "")
    else:
        ax1.scatter(X_train[i][0], X_train[i][1], color='red', alpha=0.6,
                    label='extérieur (-1)' if i == 0 else "")

cercle = plt.Circle((0, 0), np.sqrt(0.5), fill=False, color='black',
                    linestyle='--', linewidth=2, label='frontière réelle')
ax1.add_patch(cercle)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title(f"Entraînement ({len(X_train)} points)")
ax1.legend(loc='upper right', fontsize=8)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# test
for i in range(len(X_test)):
    if predire(X_test[i], params) == y_test[i]:
        couleur = 'green' if y_test[i] == 1 else 'orange'
        ax2.scatter(X_test[i][0], X_test[i][1], color=couleur, alpha=0.7,
                    marker='o', label='bien classé' if i == 0 else "")
    else:
        ax2.scatter(X_test[i][0], X_test[i][1], color='black', alpha=0.8,
                    marker='x', s=80, label='mal classé' if i == 0 else "")

cercle2 = plt.Circle((0, 0), np.sqrt(0.5), fill=False, color='black',
                     linestyle='--', linewidth=2)
ax2.add_patch(cercle2)
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title(f"Test ({len(X_test)} points) — accuracy = {acc_test:.2%}")
ax2.legend(loc='upper right', fontsize=8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.suptitle("Classification cercle — Circuit quantique variationnel 2 qubits",
             fontsize=14, fontweight='bold')
plt.tight_layout()

chemin_graphique = os.path.join(dossier_sortie, "resultats.png")
fig.savefig(chemin_graphique, dpi=150, bbox_inches="tight")
print(f"Graphique sauvegardé : {chemin_graphique}")
plt.show()

print("\nFIN")