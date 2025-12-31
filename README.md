# Fausse Commune (🇬🇧)

Fausse Commune is a web game that can be played at [faussecommune.fr](https://faussecommune.fr)

## The name generator

At the root of the project is a commune name generator, based on:
- third-order Markov chains, i.e. taking into account the last three characters to generate the next one,
- a weight specific to each commune in the model training, according to its distance from a point called the “model center”
- heuristic corrections (case management, typography, removal of existing names, etc.)

This generator allows you to create names that are appropriate for a region. For example, in Brittany: Poullannalec or Saint-Hiliac-Guiler-sur-Goyen. In Alsace: Schwickerschwihr or Vignoblenbach.

## The game

The game is based on reverse engineering the generator. The aim is to find the center of the model from the names generated. There is a system of lives, scores, and high scores.


# Fausse Commune (🇫🇷)

Fausse commune est un jeu web, jouable à l'adresse [faussecommune.fr](https://faussecommune.fr)

## Le générateur de noms

À la racine du projet, il y a un générateur de noms de communes, basé sur :
- des chaînes de Markov d'ordre 3, c'est à dire tenant compte des 3 derniers caractères pour générer le suivant,
- un poids propre à chaque commune dans l'entraînement du modèle, selon sa distance à un point appelé "centre du modèle"
- des corrections heuristiques a posteriori (gestion de la casse, de la typographie, suppression des noms déjà existants, etc)

Ce générateur permet de créer des noms adaptés à une région. Par exemple, en Bretagne : Poullannalec ou Saint-Hiliac-Guiler-sur-Goyen. En Alsace : Schwickerschwihr ou Vignoblenbach.

## Le jeu

Le jeu est basé sur la rétroingéniérie du générateur. Il s'agit à partir des noms générés de trouver le centre du modèle. Il y a un système de vies, de score, de meilleur score.