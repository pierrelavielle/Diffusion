# Projet de Semestre : Modélisation Générative de Distributions 2D par Dynamique de Langevin

**Unité d'Enseignement :** Processus Stochastiques et Méthodes de Diffusion
**Sujet :** Apprentissage de la densité de probabilité d'une structure complexe (Feuille d'Érable) par diffusion stochastique.

---

## 1. Contexte Scientifique

Les modèles de diffusion probabilistes (Denoising Diffusion Probabilistic Models - DDPM) représentent l'état de l'art actuel en modélisation générative. Ils reposent sur l'inversion d'un processus de diffusion thermodynamique, modélisé par une Équation Différentielle Stochastique (EDS).

Ce projet a pour but d'implémenter un modèle "from scratch" en dimension réduite ($d=2$). L'objectif n'est pas le traitement d'image matriciel (pixels), mais l'apprentissage de la **densité spatiale** d'un nuage de points. Nous chercherons à apprendre la fonction de score $\nabla_x \log p_t(x)$ permettant de transporter une distribution gaussienne simple vers une distribution cible complexe (la forme d'une feuille d'érable).

## 2. Objectifs Techniques et Méthodologie

Le projet se décompose en quatre phases distinctes, suivant le pipeline théorique étudié en cours (notamment les supports sur la simulation de diffusion).

### Phase A : Pré-traitement et Échantillonnage de Données (Dataset)
**Objectif :** Transformer une image binaire en une distribution de points $x \in \mathbb{R}^2$.

Vous devez constituer un jeu de données d'entraînement $x_0$ représentant la distribution cible.

1.  **Source :** Utiliser une image silhouette d'une feuille d'érable (Maple Leaf).
2.  **Discrétisation (Sampling) :** Convertir l'image en un tenseur de coordonnées $(x, y)$. Pour ce faire, considérez l'intensité des pixels comme une probabilité et effectuez un tirage aléatoire de $N$ points (ex: $N=10,000$) dans les zones noires de l'image.
3.  **Normalisation :** Centrer et réduire les données pour qu'elles soient contenues dans l'intervalle $[-1, 1]$. Cela est crucial pour stabiliser l'apprentissage du réseau neuronal.

### Phase B : Modélisation du Processus Direct (Forward Process)
**Objectif :** Implémenter le noyau de transition gaussien.

Vous devez implémenter la chaîne de Markov qui détruit progressivement l'information de la feuille d'érable. Au lieu d'une simulation pas-à-pas coûteuse pour l'entraînement, vous utiliserez la propriété de fermeture des gaussiennes vue en cours (Slide 17 du support *Processus Stochastiques III*).

Pour un pas de temps $t$ arbitraire et un plan de variance défini par $\bar{\sigma}_t$ (produit cumulé des variances), la transition s'écrit :

$$x_t = \sqrt{\bar{\sigma}_t} x_0 + \sqrt{1 - \bar{\sigma}_t} \xi$$

Où $\xi \sim \mathcal{N}(0, I)$ est un bruit blanc standard bi-dimensionnel.
**Livrable intermédiaire :** Une visualisation montrant la feuille d'érable se "dissolvant" progressivement en un nuage de points gaussien lorsque $t \to T$.

### Phase C : Estimation du Score par Réseau de Neurones (Reverse Process)
**Objectif :** Entraîner un approximateur universel (MLP) à inverser le flux de diffusion.

Puisque la densité conditionnelle inverse est inconnue, nous entraînons un réseau paramétré $\theta$ pour estimer le bruit ajouté $\xi$.

1.  **Architecture du Modèle (Perceptron Multicouche) :**
    * Le réseau doit être un **MLP (Multi-Layer Perceptron)** composé de couches linéaires (`nn.Linear`) suivies d'activations non-linéaires (ReLU, SiLU ou Swish).
    * **Entrées :** Vecteur de dimension 3 : coordonnées spatiales $(x, y)$ et l'encodage du temps $t$.
    * **Sortie :** Vecteur de dimension 2 : le bruit prédit $\xi_\theta(x_t, t)$.

2.  **Procédure d'Entraînement (Algorithme 1) :**
    Implémentez strictement l'algorithme d'optimisation présenté dans le cours (Slide 23) :
    * Échantillonner un batch de points $x_0$ de la distribution cible (feuille d'érable).
    * Échantillonner un temps $t$ uniformément dans $\{1, \dots, T\}$.
    * Générer un bruit $\xi \sim \mathcal{N}(0, I)$.
    * Calculer l'échantillon bruité $x_t$ via la formule du Processus Direct.
    * Effectuer une descente de gradient pour minimiser l'Erreur Quadratique Moyenne (MSE) :
      $$Loss = || \xi - \xi_\theta(x_t, t) ||^2$$

### Phase D : Génération par Dynamique de Langevin (Sampling)
**Objectif :** Résoudre numériquement l'EDS rétrograde pour générer la forme.

Une fois le modèle entraîné, la génération consiste à simuler la trajectoire des particules depuis une distribution prior $\mathcal{N}(0, I)$ vers la distribution des données.

* **Algorithme de Génération (Algorithme 2) :**
    Implémentez le schéma d'intégration numérique vu en cours (Slide 24) :
    1.  Initialisation : $x_T \sim \mathcal{N}(0, I)$ (nuage de points aléatoire).
    2.  Boucle itérative de $t=T$ à $1$ :
        * Mettre à jour la position des points selon la formule de dynamique de Langevin discrétisée :
        $$x_{t-1} = \frac{1}{\sqrt{\sigma_t}} \left( x_t - \frac{1 - \sigma_t}{\sqrt{1 - \bar{\sigma}_t}} \xi_{\theta}(x_t, t) \right) + \beta_t z$$
    
    *Note Fondamentale :* Le terme $z \sim \mathcal{N}(0, I)$ (si $t > 1$, sinon $z=0$) est crucial. Il représente l'injection de bruit stochastique caractéristique de la dynamique de Langevin, permettant d'explorer l'espace des phases et d'assurer la diversité.

---

## 3. Livrables Attendus

Le projet sera rendu sous la forme d'un **Notebook Jupyter** (Python/PyTorch) structuré et commenté, comprenant :

1.  **Code Source :**
    * Chargement de l'image de la feuille d'érable et conversion en dataset de points.
    * Classe `DiffusionModel` contenant le MLP.
    * Boucles d'entraînement et de génération respectant les Algorithmes 1 et 2.

2.  **Visualisation de la Dynamique :**
    * **Animation ou Série Temporelle :** Affichez l'évolution du nuage de points partant du bruit ($t=T$) et se structurant progressivement pour former la feuille d'érable ($t=0$).
    * **Champ de Vecteurs :** Une visualisation 2D des vecteurs prédits $-\xi_\theta(x, t)$ sur une grille régulière à un temps $t$ fixe. Cela permet de visualiser le "gradient" qui pousse les points vers les contours de la feuille.

3.  **Analyse :**
    * Commentez l'influence du nombre de pas de temps $T$ sur la qualité de la feuille générée.
    * Expliquez théoriquement pourquoi l'ajout de bruit $z$ dans l'étape de génération est nécessaire pour une convergence correcte (lien avec la thermodynamique).

## 4. Critères d'Évaluation

* **Rigueur Mathématique :** Correspondance exacte entre le code et les formules de transition (Forward) et de génération (Reverse) définies dans le cours.
* **Architecture Adaptée :** Utilisation correcte d'un MLP (et non d'un CNN) pour traiter des données 2D.
* **Qualité de la Reconstruction :** Le nuage de points final doit reproduire fidèlement la géométrie de la feuille d'érable, y compris ses pointes et ses zones concaves.

---
*Référence de travail : Supports de cours "Processus Stochastiques et Méthodes de Diffusion".*