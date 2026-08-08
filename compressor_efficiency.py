"""
compressor_efficiency.py

Physics-Based Compressor Efficiency Model

Hidden inputs
-------------
creep strain      (from creep_model.py)
oxide thickness    x_tgo (from oxidation_model.py)

Outputs
-------
Tip clearance
Blockage fraction
Compressor efficiency

Mirrors turbine_efficiency.py's structure, extended with a second
degradation channel: turbine_efficiency.py only has creep-driven tip
clearance loss (K1, K2 terms), this adds oxidation-driven flow-passage
blockage loss (K_block term) on top, per the uniform-deposition /
diffusion-only / no-spallation assumption -- oxide growth is treated as
geometric blockage, not a separate roughness correlation.
"""

class CompressorEfficiencyModel:

    def __init__(

        self,

        blade_height,

        blade_thickness,

        initial_clearance,

        clean_efficiency,

        K1,

        K2,

        K_block,

    ):

        self.h = blade_height

        self.t_blade = blade_thickness

        self.c0 = initial_clearance

        self.eta0 = clean_efficiency

        self.K1 = K1

        self.K2 = K2

        self.K_block = K_block

    # ------------------------------------------

    def blade_growth(

        self,

        creep_strain,

    ):

        return self.h*creep_strain

    # ------------------------------------------

    def tip_clearance(

        self,

        creep_strain,

    ):

        return (

            self.c0

            -

            self.blade_growth(creep_strain)

        )

    # ------------------------------------------

    def blockage_fraction(

        self,

        x_tgo,

    ):

        return (

            2*x_tgo

            /

            self.t_blade

        )

    # ------------------------------------------

    def efficiency(

        self,

        clearance,

        x_tgo,

    ):

        delta = clearance/self.h

        blockage = self.blockage_fraction(x_tgo)

        eta = (

            self.eta0

            -

            self.K1*delta

            -

            self.K2*delta**2

            -

            self.K_block*blockage

        )

        return eta

    # ------------------------------------------

    def update(

        self,

        creep_strain,

        x_tgo,

    ):

        clearance = self.tip_clearance(

            creep_strain

        )

        eta = self.efficiency(

            clearance,

            x_tgo,

        )

        return clearance, eta
