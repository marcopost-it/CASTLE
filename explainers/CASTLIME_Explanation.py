from lime.explanation import Explanation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CASTLIME_Explanation(Explanation):
    
    def __init__(self, dataset, labels, base_explanation, pivots, pivot_names, test_instance, class_label, true_pred, feature_names, feature_order, rule, verbose = False):
        self.base_exp = base_explanation
        self.test_instance = test_instance
        self.dataset = dataset
        self.labels = labels
        self.rule = rule
        self.feature_names = feature_names
        self.feature_order = feature_order
        self.pivots = pivots
        self.class_label = class_label
        self.true_pred = true_pred
        self.pivot_names = pivot_names
        self.exp_vector = self.get_exp_vector(base_explanation, pivots, verbose)
        
    def show_in_notebook(self):
        rule = self.rule
        test_instance = self.test_instance
        feature_names = self.feature_names
        feature_order = self.feature_order
        num_features = test_instance.shape[0]
        dataset = self.dataset
        exp_vector = self.exp_vector
        width = 220
        height = 10
        
        #offset = 15
        fig, ax = plt.subplots(1,1, figsize =(14,6))

        #Removing spines and tick marks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(bottom=False, left=False, labelsize=12)
        
        ax.set_title("My prediction is " + self.class_label + " because: \n", fontsize = 24) 

        ylim = height*num_features + (height/2)*(num_features-1)
        plt.xlim(0,width+20)
        plt.ylim(0,ylim+1)

        y_bottom = 1
        print(feature_names, ' ',feature_order[::-1])
        test_instance_toplot = test_instance[feature_order[::-1]]
        feature_names_toplot = feature_names[feature_order[::-1]]
        exp_vector_toplot = exp_vector[feature_order[::-1]]
        print("features: ", feature_names_toplot)
        print("exp_vector: ", exp_vector_toplot)
        
        plt.yticks(np.arange(5, ylim, step=15),[feature_names_toplot[i] + "    " + "{:.1f}".format(test_instance_toplot[i]) for i in range(len(feature_names_toplot))], fontsize = 20)
        plt.xticks([width/2+20],['target'], fontsize = 20)

        datasetMin = np.min(dataset,axis=0)[feature_order[::-1]]
        datasetMax = np.max(dataset,axis=0)[feature_order[::-1]]
        
        rule_toplot = rule[feature_order[::-1]]

        ax.axvline(width/2+20, lw=3, color = 'black', zorder=2)

        for i in range(num_features):
            bottom_left = (20,y_bottom)
            y_bottom += 15

            bottom_dx = (bottom_left[0] + width/2,bottom_left[1])
            rule_range = (rule_toplot[i][1] - rule_toplot[i][0])
            if rule_range == 0:
                width_dx = width_sx = 0
            else:
                width_dx = (rule_toplot[i][1]-test_instance_toplot[i])/rule_range*100
                width_sx = (test_instance_toplot[i]-rule_toplot[i][0])/rule_range*100
            #width_sx = 100 - width_dx

            bottom_sx = (bottom_left[0] + width/2 - width_sx,bottom_left[1])
                        
            color_dx = '#238823' if exp_vector_toplot[i] >= 0 else '#E42531'
            color_sx = '#238823' if exp_vector_toplot[i] < 0 else '#E42531'
            ax.add_patch(Rectangle(bottom_dx, width_dx, height, alpha=1, ec="black", lw=1.0, facecolor = color_dx, zorder=1 ))
            ax.add_patch(Rectangle(bottom_sx, width_sx, height, alpha=1, ec="black", lw=1.0, facecolor = color_sx, zorder=1 ))

            ax.text(bottom_dx[0] + width_dx + 4,(5+15*i), "{:.1f}".format(rule_toplot[i][1]), fontsize = 18, va = 'center')  
            ax.text(bottom_sx[0] - 4,(5+15*i), "{:.1f}".format(rule_toplot[i][0]), fontsize = 18, ha = 'right', va='center')
            
            #if bottom_sx[0] - 4 -1 > 20:
            #    ax.text(bottom_sx[0] - 4,(5+15*i), "{:.1f}".format(rule_toplot[i][0]), fontsize = 12, ha = 'right', va='center')
            #else: 
            #    ax.text(4,(5+15*i), "{:.1f}".format(rule_toplot[i][0]), fontsize = 12, ha = 'right', va='center')
                
            #ax.text(1, (5+15*i),"{:.1f}".format(test_instance_toplot[i]), color = 'black', va='center', ha='left', fontsize=12)
            ax.plot(np.arange(10,width+20), [5+15*i]*(width+10), zorder=0.1, color = 'grey', alpha=0.3, linestyle='--')
    
    def move_along_directions(self, model, n_points = 2000):
        dataset = self.dataset
        test_instance = self.test_instance
        exp_vector = self.exp_vector
        datasetMin = np.min(dataset,axis=0)
        datasetMax = np.max(dataset,axis=0)

        dimensionality = dataset.shape[1]

        x = np.linspace(0,n_points,n_points)        
        x = np.repeat(x,dimensionality).reshape((n_points,dimensionality))
        pts_supporting = test_instance + x*exp_vector
        pts_supporting = np.where(pts_supporting >= datasetMin, pts_supporting, datasetMin)
        pts_supporting = np.where(pts_supporting <= datasetMax, pts_supporting, datasetMax)

        preds_supporting = model.predict_proba(pts_supporting)[:,self.base_exp.available_labels()[0]]

        fig, ax = plt.subplots(1,2, figsize =(16,5))
        
        ax[0].tick_params(bottom=False, labelsize=12)
        ax[1].tick_params(bottom=False, labelsize=12)
        
        ax[0].set_ylim(0,1)
        ax[1].set_ylim(0,1)
        
        ax[0].set_xticks([])
        ax[1].set_xticks([])

        stop_point = len(pts_supporting)

        xx = range(stop_point)
        yy = preds_supporting[:stop_point]
        ax[0].plot(xx,yy, color = '#238823')
        ax[0].set_title("Supporting direction")
        ax[0].set_xlabel('Points along supporting direction')
        ax[0].set_ylabel('Model prediction')

        pts_opposing = test_instance - x*exp_vector
        pts_opposing = np.where(pts_opposing >= datasetMin, pts_opposing, datasetMin)
        pts_opposing = np.where(pts_opposing <= datasetMax, pts_opposing, datasetMax)
        preds_opposing = model.predict_proba(pts_opposing)[:,self.base_exp.available_labels()[0]]
        most_opposing = pts_opposing[np.argmin(preds_opposing)]

        stop_point = len(pts_opposing)
        xx = range(stop_point)
        yy = preds_opposing[:stop_point]
        ax[1].plot(xx,yy, color = '#E42531')

        ax[1].set_title("Opposing direction")
        ax[1].set_xlabel('Points along opposing direction')
        ax[1].set_ylabel('Model prediction')  
        
        return pts_supporting, preds_supporting, pts_opposing, preds_opposing
    
    def show_scatter_plot(self, feature_1_id, feature_2_id, feature_1_range = None, feature_2_range = None):
        test_instance = self.test_instance
        exp_vector = self.exp_vector
        pivots = self.pivots
        dataset = self.dataset
        labels = self.labels
        feature_names = self.feature_names
        true_pred = self.true_pred
        pivot_names = self.pivot_names
        
        p = np.polyfit([test_instance[feature_1_id], test_instance[feature_1_id] - exp_vector[feature_1_id]], [test_instance[feature_2_id], test_instance[feature_2_id] - exp_vector[feature_2_id]], 1)

        if exp_vector[feature_1_id] < 0: 
            x1 = np.linspace(np.min(dataset[:,feature_1_id]),test_instance[feature_1_id],2)
            x2 = np.linspace(test_instance[feature_1_id], np.max(dataset[:,feature_1_id]),2)
        else:
            x1 = np.linspace(test_instance[feature_1_id], np.max(dataset[:,feature_1_id]),2)
            x2 = np.linspace(np.min(dataset[:,feature_1_id]),test_instance[feature_1_id],2)

        y1 = np.array([i*p[0] + p[1]  for i in x1])
        y2 = np.array([i*p[0] + p[1]  for i in x2])

        fig, ax = plt.subplots(1,1, figsize =(12,8))

        plt.xlabel(feature_names[feature_1_id])
        plt.ylabel(feature_names[feature_2_id])
        
        if feature_1_range is not None:
            xlimsx = test_instance[feature_1_id] - feature_1_range
            xlimdx = test_instance[feature_1_id] + feature_1_range
            plt.xlim(xlimsx,xlimdx)
        if feature_2_range is not None:
            ylimsx = test_instance[feature_2_id] - feature_2_range
            ylimdx = test_instance[feature_2_id] + feature_2_range
            plt.ylim(ylimsx,ylimdx)

        #ax.scatter(dataset[:,feature_1_id],dataset[:,feature_2_id], label = 'dataset', zorder=0, alpha = 0.5, c = labels, cmap = matplotlib.colors.ListedColormap(['blue','red']))
        
        
        cdict = {0: 'grey', 1: 'red'}
        #cdict = {0: '#ecf3fd', 1: '#fbe6e5'}
        for g in np.unique(labels):
            ix = np.where(labels == g)
            ax.scatter(dataset[:,feature_1_id][ix], dataset[:,feature_2_id][ix], c = cdict[g], label = g, alpha=1,zorder=0)

        ax.scatter(pivots[:,feature_1_id], pivots[:,feature_2_id], color='black', label = 'pivot')
        
        if (feature_1_range is None) and (feature_2_range is None):
            for i in range(len(pivot_names)):
                ax.text(pivots[i][feature_1_id], pivots[i][feature_2_id], pivot_names[i], fontsize = 12)
        else:
            for i in range(len(pivot_names)):
                if ((pivots[i][feature_1_id] >= xlimsx) and 
                    (pivots[i][feature_1_id]<=xlimdx) and 
                    (pivots[i][feature_2_id] >= ylimsx) and 
                    (pivots[i][feature_2_id] <= ylimdx)):
                    ax.text(pivots[i][feature_1_id], pivots[i][feature_2_id], pivot_names[i], fontsize = 12)

        ax.scatter(test_instance[feature_1_id], test_instance[feature_2_id], marker = '^', s = 200, alpha=1, color='yellow',zorder=3, label='target instance')
        ax.text(test_instance[feature_1_id], test_instance[feature_2_id] - 0.1*test_instance[feature_2_id], 'prediction: ' + "{:.2f}".format(true_pred), color = 'black', horizontalalignment='center', va = 'bottom', bbox=dict(facecolor='yellow', alpha=0.7))
        #ax.annotate("",
        #        xy=(test_instance[feature_1_id], test_instance[feature_2_id]), xycoords='data',
        #        xytext=(0.8, 0.8), textcoords='data',
        #        arrowprops=dict(arrowstyle="->",
        #                        connectionstyle="arc3"),
        #        )
        ax.plot(x1,y1, color='green', ls = 'dashdot', lw = 2,zorder=2, label = 'right direction')
        ax.plot(x2,y2, color='red', ls = 'dashdot', lw = 2, zorder=2, label = 'wrong direction')

        ax.legend()
    
    
    def get_exp_vector(self, base_explanation, pivots, verbose):
        num_pivots = int(len(pivots))
        exp_pivots = []
        weights_array = np.zeros(num_pivots)
        for pair in self.base_exp.local_exp[self.base_exp.available_labels()[0]]:
            pivot_idx = pair[0]
            exp_pivots.append(pivot_idx)
            weights_array[pivot_idx] = pair[1]
        
        distance_values = self.base_exp.distance_values
        components = weights_array * distance_values
        
        vectors = []
        for i in exp_pivots:
            u_dir = pivots[i] - self.test_instance
            u_amp = np.sqrt(sum(u_dir**2))
            v_amp = np.abs(components[i])
            v = v_amp * (u_dir/u_amp)
            vectors.append(v)
        vectors = np.array(vectors)

        exp_vector_P = sum(vectors[weights_array[exp_pivots] > 0])
        exp_vector_N = sum(vectors[weights_array[exp_pivots] < 0])
        exp_vector = exp_vector_P - exp_vector_N
        if verbose:
            print("WEIGHTS: ", weights_array);
            print("DISTANCES: ", distance_values);
            print("PIVOT_VECTORS_AMPs_withsign: ", components);
            print("Explanation vector: ", exp_vector);
            
        return exp_vector
    