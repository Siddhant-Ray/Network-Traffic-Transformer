/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: George F. Riley<riley@ece.gatech.edu>
//

// TODO: Update description
// ns3 - On/Off Data Source Application class
// George F. Riley, Georgia Tech, Spring 2007
// Adapted from ApplicationOnOff in GTNetS.

#ifndef CDF_APPLICATION_H
#define CDF_APPLICATION_H

#include "ns3/address.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/data-rate.h"
#include "ns3/traced-callback.h"
#include "ns3/random-variable-stream.h"

namespace ns3
{

  class Address;
  class RandomVariableStream;
  class Socket;

  /**
 * \ingroup applications 
 * \defgroup onoff CdfApplication
 *
 * This traffic generator follows an On/Off pattern: after 
 * Application::StartApplication
 * is called, "On" and "Off" states alternate. The duration of each of
 * these states is determined with the onTime and the offTime random
 * variables. During the "Off" state, no traffic is generated.
 * During the "On" state, cbr traffic is generated. This cbr traffic is
 * characterized by the specified "data rate" and "packet size".
 */
  /**
* \ingroup onoff
*
* \brief Generate traffic to a single destination according to an
*        Cdf pattern.
*
* This traffic generator follows an On/Off pattern: after
* Application::StartApplication
* is called, "On" and "Off" states alternate. The duration of each of
* these states is determined with the onTime and the offTime random
* variables. During the "Off" state, no traffic is generated.
* During the "On" state, cbr traffic is generated. This cbr traffic is
* characterized by the specified "data rate" and "packet size".
*
* Note:  When an application is started, the first packet transmission
* occurs _after_ a delay equal to (packet size/bit rate).  Note also,
* when an application transitions into an off state in between packet
* transmissions, the remaining time until when the next transmission
* would have occurred is cached and is used when the application starts
* up again.  Example:  packet size = 1000 bits, bit rate = 500 bits/sec.
* If the application is started at time 3 seconds, the first packet
* transmission will be scheduled for time 5 seconds (3 + 1000/500)
* and subsequent transmissions at 2 second intervals.  If the above
* application were instead stopped at time 4 seconds, and restarted at
* time 5.5 seconds, then the first packet would be sent at time 6.5 seconds,
* because when it was stopped at 4 seconds, there was only 1 second remaining
* until the originally scheduled transmission, and this time remaining
* information is cached and used to schedule the next transmission
* upon restarting.
*
* If the underlying socket type supports broadcast, this application
* will automatically enable the SetAllowBroadcast(true) socket option.
*/
  class CdfApplication : public Application
  {
  public:
    /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
    static TypeId GetTypeId(void);

    CdfApplication();

    virtual ~CdfApplication();

    /**
   * \brief Return a pointer to associated socket.
   * \return pointer to associated socket
   */
    Ptr<Socket> GetSocket(void) const;

    /**
  * \brief Assign a fixed random variable stream number to the random variables
  * used by this model.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
    int64_t AssignStreams(int64_t stream);

  protected:
    virtual void DoDispose(void);

  private:
    // inherited from Application base class.
    virtual void StartApplication(void); // Called at time specified by Start
    virtual void StopApplication(void);  // Called at time specified by Stop

    //helpers
    /**
   * \brief Cancel all pending events.
   */
    void CancelEvents();

    // Event handlers
    /**
   * \brief Send a packet
   */
    void SendPacket();

    Ptr<Socket> m_socket; //!< Associated socket
    Address m_peer;       //!< Peer address
    bool m_connected;     //!< True if connected
    DataRate m_rate;      //!< Rate that data is generated
    Time m_lastStartTime; //!< Time last packet sent
    EventId m_sendEvent;  //!< Event id of pending "send packet" event
    TypeId m_tid;         //!< Type of the socket used
    
    // cdf files!
    std::string m_filename;
    double m_average_size; // in bytes!
    Ptr<EmpiricalRandomVariable> m_sizeDist;
    Ptr<ExponentialRandomVariable> m_timeDist;
    uint32_t m_counter;   // track number of fragments sent

    /// Traced Callback: transmitted packets.
    TracedCallback<Ptr<const Packet>> m_txTrace;

    /// Callbacks for tracing the packet Tx events, includes source and destination addresses
    TracedCallback<Ptr<const Packet>, const Address &, const Address &> m_txTraceWithAddresses;

  private:
    /**
   * \brief Schedule the next packet transmission
   */
    void ScheduleNextTx();
    /**
   * \brief Handle a Connection Succeed event
   * \param socket the connected socket
   */
    void ConnectionSucceeded(Ptr<Socket> socket);
    /**
   * \brief Handle a Connection Failed event
   * \param socket the not connected socket
   */
    void ConnectionFailed(Ptr<Socket> socket);

    // Accessors for Distribution Attributes
    bool SetDistribution(std::string filename);
    std::string GetDistribution() const;

    void SetRate(DataRate rate);
    DataRate GetRate() const;

    // Helper to set the rate dist, needs to be called by both setters above.
    void UpdateRateDistribution();
  };

} // namespace ns3

#endif /* CDF_APPLICATION_H */
