{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMupv+4PD5xj+PyjyKbca7e"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "OAl8BMOxX6Tg"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import numpy"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cel = numpy.array([-40,-10,0,8,15,22,38], dtype=float)\n",
        "far = numpy.array([-40,14,32,46,59,72,100], dtype=float)\n",
        "\n",
        "for i,e in enumerate(cel):\n",
        "  print(\"{} degree clesius = {} degree fahrenheit\".format(e, far[i]))"
      ],
      "metadata": {
        "id": "p9k6dqI1aK0e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = tf.keras.Sequential([tf.keras.layers.Dense(units = 14,input_shape = [1]),tf.keras.layers.Dense(units = 4),tf.keras.layers.Dense(units = 1)])"
      ],
      "metadata": {
        "id": "G0c8J6iQcjms"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))"
      ],
      "metadata": {
        "id": "XuR1LL4Wevfz"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history=model.fit(cel,far,epochs=2000,verbose=False)"
      ],
      "metadata": {
        "id": "HZuPYrFGfK4R"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(model.predict([100]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aHNJHZJFfno4",
        "outputId": "9fdb392b-13f2-481f-8066-c5dbde937549"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[211.74667]]\n"
          ]
        }
      ]
    }
  ]
}