Attribute
=====

The Per-Unit Power System Representation:

    - ``self.MVA_base`` : Power base (MW)

    - ``self.KV_base``  : Voltage base (kV)

    - ``self.Z_base``   : Impedance base 

    - ``self.Y_base``   : Admipedance base

    - ``self.total_kW_demand`` : Total active power (demand)

    - ``self.total_kVAr_demand``: Total reactive power (demand)

    - ``self.total_kW_supply`` : Total active power (supply)

    - ``self.total_kVAr_supply``: Total reactive power (supply)


For any the ``i`` *th* bus,

    - ``self.S[i]`` : a complex power at the ``i`` *th* bus

    - ``self.V[i]`` : a complex voltage at the ``i`` *th* bus

    - ``self.Psi_k[i]`` : a complex voltage product function at the ``i`` *th* bus
 
    - ``self.Ybus`` : a complex matrix addmittant



