---
topic: boat
updated: 2026-03-13
---

# MMSI & AIS Registration — 2011 World Cat 290 DC

> Reference guide for getting the Standard Horizon VHF radio and Garmin GPSMAP 1643xsv properly registered and programmed with an MMSI number. This is on the boat maintenance punch list.

---

## What Is MMSI?

A **Maritime Mobile Service Identity (MMSI)** is a unique 9-digit number permanently assigned to your vessel — think of it as a "phone number" for your boat's radio equipment.

Your MMSI is used by two critical safety systems:

| System | What It Does |
|--------|-------------|
| **DSC (Digital Selective Calling)** | When you press MAYDAY on your VHF, it transmits a digital distress burst containing your MMSI, GPS position, and timestamp to the Coast Guard and nearby vessels |
| **AIS (Automatic Identification System)** | If you add a Class B AIS transponder, it broadcasts your vessel identity, position, course, and speed using the same MMSI |

**Without a programmed MMSI**, the DSC distress button on the Standard Horizon radio cannot properly alert the Coast Guard with your identity or position — the signal goes out but contains no identifying information.

U.S. recreational vessel MMSIs always begin with **338** (U.S. country code).

---

## Your Vessel Details (for Registration Form)

| Field | Value |
|-------|-------|
| **Vessel Name** | *(confirm — what name is on the boat?)* |
| **Make / Model** | 2011 World Cat 290 DC |
| **Hull ID (HIN)** | EPY03172J011 |
| **State Registration** | MA 8259BH |
| **Length** | 29 ft |
| **Vessel Type** | Power catamaran |
| **Engine Type** | Twin outboard |
| **Fuel Type** | Gasoline |
| **Chartplotter** | Garmin GPSMAP 1643xsv (S/N: 8GW001105) |
| **VHF Radio** | Standard Horizon *(confirm model — likely GX series)* |

---

## Step 1 — Register Your MMSI

### Recommended: BoatUS Foundation (Free, Immediate)

**For domestic U.S. use** (Massachusetts, New England, offshore U.S. waters):

1. Go to **[boatus.com/mmsi](https://www.boatus.com/mmsi)**
2. Click **"Register Now"** → Select **"Recreational Vessel"**
3. Fill in vessel info (table above) + owner contact info + emergency contacts
4. Submit — your **9-digit MMSI is issued immediately**
5. Print or save the confirmation

**What you'll need to provide:**
- All vessel info above
- Your legal name, address, phone, email
- **Emergency contact** (someone NOT aboard — name, phone, relationship)
- VHF radio make/model, GPS make/model
- EPIRB hex ID if you have one

**Alternative:** Sea Tow ([seatow.com/boating-safety/mmsi](https://www.seatow.com/boating-safety/mmsi)) — equally valid, same Coast Guard database.

---

### If You Travel to Canada or International Waters → FCC License Required

If you go to Canadian waters (common from New England), you also need an **FCC Ship Station License**:

- **Website:** [wireless.fcc.gov/uls](https://wireless.fcc.gov/uls)
- **Cost:** ~$35 / 10-year license
- **Service type:** Maritime Mobile (Ship Station)
- Register with BoatUS first to get your MMSI, then include that same MMSI on the FCC application
- Also requires a **Restricted Radiotelephone Operator Permit (MROP)** for the operator

> **Note:** You can only have ONE MMSI per vessel. BoatUS and FCC use the same 9-digit number — get one and keep it.

---

## Step 2 — Program MMSI into the Standard Horizon VHF Radio

> ⚠️ **Critical:** The MMSI can only be entered **once** on Standard Horizon radios. It cannot be changed by the user after programming. **Triple-check your number before entering it.** If you make a mistake, contact Standard Horizon service for a factory reset.

### General Steps for Standard Horizon GX Series

1. **Power on** the radio in normal operating mode
2. Press **[MENU]** (or hold **[CALL/CH16]** depending on model)
3. Navigate to **DSC** → **DSC Setup** → **MMSI**
4. Radio will warn: *"MMSI can only be set once — proceed?"* → Confirm
5. **Enter your 9 digits** carefully using the keypad/rotary selector
6. **Confirm entry** when prompted — radio saves and locks the MMSI
7. Verify: go back to DSC menu and confirm MMSI is displayed correctly

**For your exact model's button sequence,** download the manual at:
→ **[standardhorizon.com](https://www.standardhorizon.com)** → Support → Download Manuals → find your GX model

### Connect GPS to VHF (Essential for DSC)

Your Garmin GPSMAP 1643xsv supports **NMEA 2000**, and newer Standard Horizon GX radios also support NMEA 2000. Confirm the connection type for your specific radio:

| Connection Type | What to Do |
|----------------|-----------|
| **NMEA 2000** (preferred) | Both devices share the boat's N2K backbone — they auto-communicate once connected |
| **NMEA 0183** | 4-wire connection: Radio NMEA IN ← GPS NMEA OUT (TX+/TX-/RX+/RX-) |

**Verify GPS link is working:**
On the VHF radio → DSC menu → check that the display shows a live lat/lon position (NOT "No GPS" or "Position Unknown"). If it shows a position, your MAYDAY button will transmit your location.

> **AMG Marine (Aris)** at your Amesbury storage facility handled your electronics install — he can confirm how the VHF and chartplotter are connected and verify the NMEA interface. Contact: aris.c@amg-marine.com / (617) 592-9359.

---

## Step 3 — Program MMSI into the Garmin GPSMAP 1643xsv

The Garmin 1643xsv does not have a built-in VHF radio, so the MMSI lives in the Standard Horizon radio. However, the Garmin can be configured to:
- Display incoming DSC distress calls from other vessels on the chart
- Show your vessel's identity in DSC contact data
- Log your own MMSI for reference

**If you want to set MMSI in the Garmin for contact/display purposes:**
1. Home screen → **Settings** (gear icon)
2. Navigate to **Communications** → **DSC**
3. Enter your MMSI in the **My MMSI** field

**For the Garmin to display AIS targets** (other vessels' AIS broadcasts):
- The 1643xsv may have a built-in AIS receiver — check your unit's specs under **Settings → System → About** or in the product spec sheet at [support.garmin.com](https://support.garmin.com)
- If AIS is available, it will auto-populate AIS targets on the chart with no configuration required (it's receive-only, no MMSI needed for receive)
- If you add a Class B AIS transponder later, use the same MMSI and connect it to the Garmin via NMEA 2000

---

## AIS vs. MMSI — Clarification

| Term | Meaning | Registration Needed? |
|------|---------|---------------------|
| **MMSI** | The 9-digit vessel identity number | Yes — via BoatUS or FCC (see above) |
| **DSC** | Digital distress/call function built into VHF radios | Uses MMSI — no separate registration |
| **AIS Receiver** | Receive-only — sees other vessels' broadcasts | No registration needed |
| **Class B AIS Transponder** | Broadcasts your position to other vessels | Uses same MMSI — no separate registration |

> There is no separate "AIS registration" in the U.S. Your MMSI registration (BoatUS/FCC) covers all uses.

---

## Step 4 — Verify and Test

After registration and programming:

1. **Confirm MMSI in database:** Go to [navcen.uscg.gov](https://navcen.uscg.gov) and search your MMSI — your vessel details should appear
2. **Verify GPS link on VHF:** Check DSC status screen on the radio — should show live position
3. **Test DSC call (non-emergency):** You can send a routine DSC individual position report to another vessel or marina (NOT the red MAYDAY button — that alerts the Coast Guard)
4. **Keep registration updated:** If vessel name, owner info, or emergency contacts change, update at boatus.com/mmsi

---

## EPIRB Note

If you add an EPIRB to the boat (strongly recommended for offshore):
- Register it separately at **[beaconregistration.noaa.gov](https://beaconregistration.noaa.gov)**
- Include your MMSI in the EPIRB registration to link them in the SAR database
- EPIRB registration is free and separate from VHF MMSI registration

---

## Key URLs

| Resource | URL |
|----------|-----|
| BoatUS MMSI Registration (free) | [boatus.com/mmsi](https://www.boatus.com/mmsi) |
| Sea Tow MMSI Registration (free) | [seatow.com/boating-safety/mmsi](https://www.seatow.com/boating-safety/mmsi) |
| FCC Ship Station License | [wireless.fcc.gov/uls](https://wireless.fcc.gov/uls) |
| USCG NAVCEN — Verify MMSI | [navcen.uscg.gov](https://navcen.uscg.gov) |
| NOAA EPIRB Registration | [beaconregistration.noaa.gov](https://beaconregistration.noaa.gov) |
| Standard Horizon Manuals | [standardhorizon.com](https://www.standardhorizon.com) → Support → Manuals |
| Garmin Support | [support.garmin.com](https://support.garmin.com) |

---

## Contacts (If You Need Help)

| Contact | Role | Details |
|---------|------|---------|
| **Aris — AMG Marine** | Electronics installer (Amesbury, MA) | aris.c@amg-marine.com / (617) 592-9359 — at NE Auto & Boat Storage |
| **Standard Horizon Support** | Radio factory MMSI reset if needed | standardhorizon.com → Contact |

---

## Status

- [ ] Confirm Standard Horizon VHF model number
- [ ] Register MMSI at boatus.com/mmsi
- [ ] Program MMSI into Standard Horizon VHF
- [ ] Verify GPS (Garmin 1643xsv) is feeding position to VHF over NMEA 2000
- [ ] Confirm MMSI in USCG database at navcen.uscg.gov
- [ ] Decide: FCC Ship Station License needed? (only if traveling to Canada/international)
- [ ] Check if Garmin 1643xsv has built-in AIS receiver (Settings → System → About)
