import sys,random,time,math
import numpy as np
from numpy.random import RandomState
from random import shuffle
from Tkinter import *
import time
import random
from math import *
import sys

#FFNN supervised learning packages 
# from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
# from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
# from pybrain.tools.validation import ModuleValidator

# Author : Nguyen Sao Mai
# nguyensmai@gmail.com
# nguyensmai.free.fr

class World(Frame): 
    couleurs = ['cornflower blue', 'slate blue', 'cadet blue','aquamarine','light slate blue', 'medium blue', 'blue', 'midnight blue', 'turquoise','deep sky blue','cyan']    

    # constructor: initialises a world of size larg x haut with nboules balls
    def __init__(self, master=None, larg=300, haut=300, nboules=5, tempo=0.0005):
        Frame.__init__(self, master)

        #sauvegarde des parametres
        self.master = master
        self.larg = larg
        self.haut = haut
        self.nboules = nboules
        self.tempo = tempo
        self.state_size=5*nboules-2 # states are (x,y,dx,dy,r) for the 5 balls except for boules[0] (dx,dy are given then by m)
        self.action_size=2          # motor command is the velocity [dx,dy] of boules[0]
        #Create world data structures 
        self.s = [0]*(self.state_size-2)    #CURRENT STATE 
        self.stp1 = [0]*(self.state_size-2) #TEMPORARY STATE. 
        self.m = [];                        #last motor command
        self.boules = []                    # the actual states of the balls

        # creation of a layout_manager
        self.grid()
 
        # creation of the canvas to plot the world and the balls
        self.can = Canvas(self, width=self.larg, height=self.haut, bg='ivory')
        self.can.grid(sticky="nsew")
        self.resetState()

 



    # reset the balls to new states (new positions, velocities and radius)
    def resetState(self):
        #clear the canvas
        self.can.delete(ALL)
        self.boules = []

        for i in xrange(0,self.nboules):
            if i==0:
                r=30;
            else:
                r=random.randint(15,30)  # = ball radius

            dz = random.uniform(0.5,5)  # = velocity magnitude
            c = random.uniform(0,2*pi)  # = velocity direction ( 0 to 2*pi radians)
            dy, dx = dz*sin(c), dz*cos(c)  # = velocity vector along directions x and y
            # find a position for the ball which does not overlap with the previous balls' positions
            while True:
                x=random.randint(r,self.larg-r)
                y=random.randint(r,self.haut-r)
                ok=True
                for boule2,x2,y2,dx2,dy2,r2 in self.boules:
                    if sqrt((x2-x)**2+(y2-y)**2)<r+r2:
                        ok=False
                if ok:
                    break
            # plotting the balls
            # attention: changement d'ordonnees necessaire: dans le canevas, l'axe d'ordonnee va vers le bas
            # en plus: (y,x) est le centre de la boule, mais les parametres a passer sont differents
            if i==0:
                boule = self.can.create_oval(x-r, self.haut-(y+r), x+r, self.haut-(y-r), width=2, fill='red')
            else:
                boule = self.can.create_oval(x-r, self.haut-(y+r), x+r, self.haut-(y-r), width=2, fill=World.couleurs[i])
 
            # add this new ball to the list of balls
            self.boules.append([boule,x,y,dx,dy,r])
            if i==0:
                self.s[i*5:(i+1)*5-2] = [x/self.larg,y/self.haut,r/30.]
            else:
                self.s[i*5-2:(i+1)*5-2] = [x/self.larg,y/self.haut,dx/5.,dy/5.,r/30.]

        # print "RESETSTATE"
        # print len(self.s)
        return self.s



    # plotting when we test the expected balls and the predicted balls. plots is a vector of [predicted[i],[expected[i]]
    def plotPrediction(self, plots):
        self.can.delete(ALL)
        dimmax = len(plots)
        tmax = len(plots[1])
        for t in range(tmax): #time
            #boule 0
            i=0;
            [xp, xe]= plots[i,t]*self.larg
            i=1;
            [yp, ye]= plots[i,t]*self.haut
            i=2;
            [rp, re]= plots[i,t]*30.
            # expected ball
            boule = self.can.create_oval(xe-re, self.haut-(ye+re), xe+re, self.haut-(ye-re), width=2, fill='red')
            # predicted ball
            boule = self.can.create_oval(xp-rp, self.haut-(yp+rp), xp+rp, self.haut-(yp-rp), width=4, outline='red')

            for iboule in range(self.nboules-1):
                #boule iboule 
                i=1+2+5*iboule;
                [xp, xe]= plots[i,t]*self.larg
                i=2+2+5*iboule;
                [yp, ye]= plots[i,t]*self.haut
                i=5+2+5*iboule;
                [rp, re]= plots[i,t]*30.
                boule = self.can.create_oval(xe-re, self.haut-(ye+re), xe+re, self.haut-(ye-re), width=2, fill=World.couleurs[iboule])
                boule = self.can.create_oval(xp-rp, self.haut-(yp+rp), xp+rp, self.haut-(yp-rp), width=4, outline=World.couleurs[iboule])

            # refresh display
            time.sleep(0.5)
            self.master.update()
            self.can.delete(ALL)

           

    #computes the direction of the velocity vector
    def direction(self, dy, dx):
        """donne l'angle de 0 a 2*pi donne par les increments (signes) dy et dx"""
        a = atan(abs(dy/dx))
        if dy>=0:
            if dx>=0:
                pass
            else:
                a = pi-a
        else:
            if dx>=0:
                a = 2*pi-a
            else:
                a = pi+a
        return a



    def updateState(self, m, plot=1):
        self.s =self.stp1;
        try:
            # for each ball
            for i,(boule,x,y,dx,dy,r) in zip(range(0,len(self.boules)),self.boules):
        
                #  compute its next position
                if i==0:
                    [dx, dy] = m

                x, y = x+dx, y+dy
        
                # changes the trajectory if collision with other balls
                for j,(boule2,x2,y2,dx2,dy2,r2) in zip(range(0,len(self.boules)),self.boules):
                    if j==0:
                        [dx2, dy2] = m

                    if j!=i:
        
                        # calcul de la distance entre les centres des 2 boules i et j
                        db=sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y))
        
                        # test pour savoir s'il y a collision avec une autre boule
                        if db<=r+r2:
                            # oui, il y a collision de la boule i avec la boule j
        
                            # recul de la boule i pour ne pas avoir de recouvrement avec la boule j
                            x, y = x-dx, y-dy
        
                            # direction prise actuellement par la boule i (de 0 a 2*pi radians)
                            a = self.direction(dy,dx)
        
                            # direction donnee par la ligne des 2 boules a partir de la boule i (de 0 a 2*pi radians)
                            b = self.direction(y2-y, x2-x)
        
                            # nouvelle direction que doit prendre la boule i apres rebond sur la boule j (de 0 a 2*pi radians)
                            c = (2*a-b+(2*pi)) % (2*pi)
        
                            # calcul du nouvel increment dx et dy de la boule i selon la nouvelle direction c
                            dz = sqrt(dy*dy+dx*dx)
                            dy, dx = dz*sin(c), dz*cos(c)
        
                # corrige la trajectoire en fonction des collisions avec les bords
                if x>self.larg-r:
                    x = self.larg-r
                    dx = -abs(dx)
                if y>self.haut-r:
                    y = self.haut-r
                    dy = -abs(dy)
                if x<r:
                    x = r
                    dx = abs(dx)
                if y<r:
                    y = r
                    dy = abs(dy)
        
                # dessine la boule a la nouvelle position (voir remarque plus haut concernant le dessin des boules)
                if plot==1:
                    self.can.coords(boule, x-r, self.haut-(y+r), x+r, self.haut-(y-r))
        
                # enregistre les nouvelles donnees de la boule i
                self.boules[i]=[boule,x,y,dx,dy,r]
                if i==0:
                    self.stp1[i*5:(i+1)*5-2] = [x/self.larg,y/self.haut,r/30.]
                    self.m= m;
                else:
                    self.stp1[i*5-2:(i+1)*5-2] = [x/self.larg,y/self.haut,dx/5.,dy/5.,r/30.]

            #plotting
            if plot==1:
                # temporisation pour ralentir l'affichage
                time.sleep(self.tempo)
                # rafraichir l'affichage
                self.master.update()

            return self.stp1
        except:
            print "WORLD:UPDATESTATE error"





    def getState(self):
        return self.s

    
    def getRandomMotor(self):
        # generates only a new random motor command once every 5 steps, to keep smooth movements
        if random.uniform(0,1)<0.2 or not self.m:
            dz = random.uniform(0.5,5)  # = pas d'avancement de la boule
            c = random.uniform(0,2*pi)  # = direction d'avancement (de 0 a 2*pi radians)
            dy, dx = dz*sin(c), dz*cos(c)  # = increments d'avancement selon x et y
        else:
            [dx, dy] = self.m 

        self.m= [dx,dy];
        return [dx,dy]
    
