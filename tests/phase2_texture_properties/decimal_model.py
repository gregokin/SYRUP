"""AUTHOR-written Decimal80 transcription of the provider recipe INCLUDING measured anchors/overrides (Root's portable controls in ./oracles cover the no-override branches independently). This model
shares the author's reading of the recipe, so it is a cross-check of the implementation arithmetic, not an independent derivation."""
from __future__ import annotations

from decimal import Decimal as D
from decimal import localcontext

TWO_65 = D("2.65")


def model(S, Si, C, OM, rho_kg, wp_o=None, f33_o=None, sat_o=None, air_o=None, ks_o=None):
    """Returns (flags set of names, values dict). Decimal80; inputs are floats or strings."""
    with localcontext() as ctx:
        ctx.prec = 80
        S, Si, C, OM, rho = D(S), D(Si), D(C), D(OM), D(rho_kg) / 1000
        tw = -D(".024") * S + D(".487") * C + D(".006") * OM + D(".005") * S * OM - D(".013") * C * OM + D(".068") * S * C + D(".031")
        wpn = tw + D(".14") * tw - D(".02")
        tf = -D(".251") * S + D(".195") * C + D(".011") * OM + D(".006") * S * OM - D(".027") * C * OM + D(".452") * S * C + D(".299")
        fcn = tf + D("1.283") * tf * tf - D(".374") * tf - D(".015")
        tg = D(".278") * S + D(".034") * C + D(".022") * OM - D(".018") * S * OM - D(".027") * C * OM - D(".584") * S * C + D(".078")
        g = tg + D(".636") * tg - D(".107")
        satn = fcn + g - D(".097") * S + D(".043")
        et = -D("21.67") * S - D("27.93") * C - D("81.97") * g + D("71.12") * S * g + D("8.29") * C * g + D("14.05") * S * C + D("27.16")
        airn = et + D(".02") * et * et - D(".113") * et - D(".70")
        rhon = (1 - satn) * TWO_65
        df = rho / rhon
        satd = 1 - rho / TWO_65
        fcd = fcn - D(".2") * (satn - satd)
        wp = wpn if wp_o is None else D(wp_o)
        f33 = fcd if f33_o is None else D(f33_o)
        sat = satd if sat_o is None else D(sat_o)
        air = airn if air_o is None else D(air_o)
        gap = sat - f33
        flags = set()
        if not (D("0.9") <= df <= D("1.3")):
            flags.add("df")
        if not (0 < wp < f33 < sat < 1):
            flags.add("curve")
        if not (0 <= air < 33):
            flags.add("air")
        if not gap >= D("0.005"):
            flags.add("gap")
        vals = dict(wp=wp, f33=f33, sat=sat, air=air, df=df, gap=gap)
        if not flags:
            B = (D(1500).ln() - D(33).ln()) / (f33.ln() - wp.ln())
            lam = 1 / B
            ks = D(1930) * ((3 - lam) * gap.ln()).exp() / 3600000 if ks_o is None else D(ks_o)
            eq = D(6) <= air
            fc6 = sat if eq else f33 + D(27) * gap / (D(33) - air)
            vals.update(B=B, lam=lam, ks=ks, fc6=fc6, eq=eq)
        return flags, vals
