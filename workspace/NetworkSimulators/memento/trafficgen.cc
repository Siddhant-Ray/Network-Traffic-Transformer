/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

// Network topology
//
//                               disturbance
//                                  |
//        sender --- switchA --- switchB --- receiver
//
//  The "disturbance" host is used to introduce changes in the network
//  conditions.
//

#include <iostream>
#include <fstream>
#include <unordered_map>

#include <chrono>
#include <ctime>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/gnuplot.h"
#include "ns3/node.h"
#include "ns3/traffic-control-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/ipv4-header.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"


#include "ns3/cdf-application.h"
#include "ns3/experiment-tags.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TrafficGenerationExperiment");

const auto TCP = TypeIdValue(TcpSocketFactory::GetTypeId());
const auto UDP = TypeIdValue(UdpSocketFactory::GetTypeId());

// Put the current timestamp and packet size into a log.
void logSize(Ptr<OutputStreamWrapper> stream, Ptr<Packet const> p)
{
    auto current_time = Simulator::Now();
    *stream->GetStream() << current_time.GetSeconds() << ','
                         << p->GetSize() << std::endl;
}

// Tag a packet with a timestamp.
void setTimeTag(Ptr<Packet const> p)
{
    TimestampTag tag;
    tag.SetTime(Simulator::Now());
    p->AddPacketTag(tag);
};

void setIdTag(u_int32_t workload_id, u_int32_t app_id, Ptr<Packet const> p)
{
    IdTag tag;
    tag.SetWorkload(workload_id);
    tag.SetApplication(app_id);
    p->AddPacketTag(tag);
};

// Log workload tag, timestamp tag, and packet size.
void logPacketInfo(Ptr<OutputStreamWrapper> stream, Ptr<Packet const> p)
{
    TimestampTag timestampTag;
    IdTag idTag;
    FlowIdTag flowid;
    if (p->PeekPacketTag(timestampTag) && p->PeekPacketTag(idTag))
    {
        auto current_time = Simulator::Now();
        auto diff_time = current_time - timestampTag.GetTime();
        *stream->GetStream() << "Tx sent at:, "<< current_time.GetSeconds()<< ", ";
        *stream->GetStream() << "Flow id is, "<< p->PeekPacketTag(flowid) << ", "
                             << "Packet uid is, "<< p->GetUid() << ", "
                             << "Packet size is, "<< p->GetSize() << ", ";
                             
        Ptr<Packet> copy = p->Copy();
        // Headers must be removed in the order they're present.
        EthernetHeader eHeader;
        copy->RemoveHeader(eHeader);
        Ipv4Header ipHeader;
        copy->RemoveHeader(ipHeader);
        *stream->GetStream() << "IP ID is, "<< ipHeader.GetIdentification() << ", "
                                << "DSCP is, "<< ipHeader.GetDscp() << ", "
                                << "ECN is, "<< ipHeader.GetEcn() << ", "
                                << "TTL is, "<< 64 << ", "
                                << "Payload size is, "<< ipHeader.GetPayloadSize() << ", "
                                << "Protocol is, "<< 6 << ", "
                                << "Source IP is, "<< ipHeader.GetSource() << ", "
                                << "Destination IP is, "<< ipHeader.GetDestination() << ", ";
        TcpHeader tcpHeader;
        copy->RemoveHeader(tcpHeader);
        *stream->GetStream() << "TCP source port is, "<< tcpHeader.GetSourcePort() << ", "
                             << "TCP destination port is, "<< tcpHeader.GetDestinationPort() << ", "
                             << "TCP sequence num is, "<< tcpHeader.GetSequenceNumber() << ", "
                             << "TCP current window size is, "<< tcpHeader.GetWindowSize() << ", "
                             << "Delay is, "<< diff_time.GetSeconds() << ", "
                             << "Workload id is, "<< idTag.GetWorkload() << ','
                             << "Application id is, "<< idTag.GetApplication() << ',';

        copy->Print(*stream->GetStream());
        *stream->GetStream() << "\n";

    }
    else
    {
        //NS_LOG_DEBUG("Packet without timestamp, won't log.");
    };
};

// TODO: Add base stream? Or how to get different random streams?
Ptr<RandomVariableStream> TimeStream(double min = 0.0, double max = 1.0)
{
    return CreateObjectWithAttributes<UniformRandomVariable>(
        "Min", DoubleValue(min),
        "Max", DoubleValue(max));
}

PointerValue TimeStreamValue(double min = 0.0, double max = 1.0)
{
    return PointerValue(TimeStream(min, max));
}

NetDeviceContainer GetNetDevices(Ptr<Node> node)
{
    NetDeviceContainer devices;
    for (uint32_t i = 0; i < node->GetNDevices(); ++i)
    {
        devices.Add(node->GetDevice(i));
    }
    return devices;
}

int main(int argc, char *argv[])
{   
    // Measure wall clock time 
    auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	
	std::cout << "Started computation at " << std::ctime(&start_time);
    //
    // Users may find it convenient to turn on explicit debugging
    // for selected modules; the below lines suggest how to do this
    //
    #if 1
    LogComponentEnable("TrafficGenerationExperiment", LOG_LEVEL_INFO);
    #endif
    //
    // Allow the user to override any of the defaults and the above Bind() at
    // run-time, via command-line arguments
    //

    int n_apps = 10;
    DataRate linkrate("5Mbps");
    DataRate baserate("100kbps");
    DataRate congestion("0Mbps");
    std::string basedir = "./distributions/";
    std::string w1 = basedir + "Facebook_WebServerDist_IntraCluster.txt";
    std::string w2 = basedir + "DCTCP_MsgSizeDist.txt";
    std::string w3 = basedir + "Facebook_HadoopDist_All.txt";
    std::string prefix = "shift";
    double c_w1 = 1;
    double c_w2 = 1;
    double c_w3 = 1;

    CommandLine cmd;
    cmd.AddValue("apps", "Number of traffic apps per workload.", n_apps);
    cmd.AddValue("apprate", "Base traffic rate for each app.", baserate);
    cmd.AddValue("linkrate", "Link capacity rate.", linkrate);
    cmd.AddValue("w1", "Factor for W1 traffic (FB webserver).", c_w1);
    cmd.AddValue("w2", "Factor for W2 traffic (DCTCP messages).", c_w2);
    cmd.AddValue("w3", "Factor for W3 traffic (FB hadoop).", c_w3);
    cmd.AddValue("congestion", "Congestion traffic rate.", congestion);
    cmd.AddValue("prefix", "Prefix for log files.", prefix);
    cmd.Parse(argc, argv);

    // Compute resulting workload datarates.
    auto rate_w1 = DataRate(static_cast<uint64_t>(c_w1 * baserate.GetBitRate()));
    auto rate_w2 = DataRate(static_cast<uint64_t>(c_w2 * baserate.GetBitRate()));
    auto rate_w3 = DataRate(static_cast<uint64_t>(c_w3 * baserate.GetBitRate()));

    // Print Overview of seetings
    NS_LOG_DEBUG("Overview:"
                 << std::endl
                 << "Congestion: "
                 << congestion << std::endl
                 << "Apps: "
                 << n_apps << std::endl
                 << "Workloads (data rates per app):"
                 << std::endl
                 << "W1: " << w1 << " (" << rate_w1 << ")"
                 << std::endl
                 << "W2: " << w2 << " (" << rate_w2 << ")"
                 << std::endl
                 << "W3: " << w3 << " (" << rate_w3 << ")");

    // Simulation variables
    auto simStart = TimeValue(Seconds(0));
    auto stopTime = Seconds(10);
    auto simStop = TimeValue(stopTime);

    // Fix MTU and Segment size, otherwise the small TCP default (536) is used.
    Config::SetDefault("ns3::CsmaNetDevice::Mtu", UintegerValue(1500));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1380));

    //
    // Explicitly create the nodes required by the topology (shown above).
    //

    // Network topology
    //
    //                               disturbance
    //                                  |
    //        sender --- switchA --- switchB --- receiver
    //                      |
    //                   switchC --- switchD --- (receiver)
    //
    //
    //
    //
    //
    //  The "disturbance" host is used to introduce changes in the network
    //  conditions.
    //


    NS_LOG_INFO("Create nodes (hosts and disturbances).");
    NodeContainer hosts;
    hosts.Create(3);
    // Keep references to sender, receiver, and disturbance
    auto sender = hosts.Get(0);
    auto receiver = hosts.Get(1);
    auto disturbance = hosts.Get(2);

    NodeContainer switches;
    switches.Create(4);
    auto switchA = switches.Get(0);
    auto switchB = switches.Get(1);
    auto switchC = switches.Get(2);
    auto switchD = switches.Get(3);
    

    NS_LOG_INFO("Build Topology");
    CsmaHelper csma;
    csma.SetChannelAttribute("FullDuplex", BooleanValue(true));
    csma.SetChannelAttribute("DataRate", DataRateValue(linkrate));
    csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(5)));

    // Create the csma links
    csma.Install(NodeContainer(sender, switchA));
    csma.Install(NodeContainer(receiver, switchB));
    csma.Install(NodeContainer(disturbance, switchB));
    csma.Install(NodeContainer(switchA, switchB));
    csma.Install(NodeContainer(switchA, switchC));
    csma.Install(NodeContainer(switchC, switchD));

    // Create the bridge netdevice, turning the nodes into actual switches
    BridgeHelper bridge;
    bridge.Install(switchA, GetNetDevices(switchA));
    bridge.Install(switchB, GetNetDevices(switchB));

    bridge.Install(switchC, GetNetDevices(switchC));
    bridge.Install(switchD, GetNetDevices(switchD));

    // Add internet stack and IP addresses to the hosts
    NS_LOG_INFO("Setup stack and assign IP Addresses.");
    NetDeviceContainer hostDevices;
    hostDevices.Add(GetNetDevices(sender));
    hostDevices.Add(GetNetDevices(receiver));
    hostDevices.Add(GetNetDevices(disturbance));

    InternetStackHelper internet;
    internet.Install(hosts);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    auto addresses = ipv4.Assign(hostDevices);
    // Get Address: Device with index 0/1, address 0 (only one address)
    auto addrReceiver = addresses.GetAddress(1, 0);

    NS_LOG_INFO("Create Traffic Applications.");
    uint16_t base_port = 4200; // Note: We need two ports per pair
    auto trafficStart = TimeStream(1, 2);
    for (auto i_app = 0; i_app < n_apps; ++i_app)
    {
        // We also need to set the appropriate tag at every application!

        // Addresses
        auto port = base_port + i_app;
        auto recvAddr = AddressValue(InetSocketAddress(addrReceiver, port));

        // Sink
        Ptr<Application> sink = CreateObjectWithAttributes<PacketSink>(
            "Local", recvAddr, "Protocol", TCP,
            "StartTime", simStart, "StopTime", simStop);
        receiver->AddApplication(sink);

        // Sources for each workload
        // App indexing scheme: 0--n_apps-1: w1, n_apps -- 2n_apps-1: w2, etc.
        if (rate_w1 > 0)
        {
            Ptr<CdfApplication> source1 = CreateObjectWithAttributes<CdfApplication>(
                "Remote", recvAddr, "Protocol", TCP,
                "DataRate", DataRateValue(rate_w1), "CdfFile", StringValue(w1),
                "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
                "StopTime", simStop);
            source1->TraceConnectWithoutContext(
                "Tx", MakeBoundCallback(&setIdTag, 1, i_app));
            sender->AddApplication(source1);
        }
        if (rate_w2 > 0)
        {
            Ptr<CdfApplication> source2 = CreateObjectWithAttributes<CdfApplication>(
                "Remote", recvAddr, "Protocol", TCP,
                "DataRate", DataRateValue(rate_w2), "CdfFile", StringValue(w2),
                "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
                "StopTime", simStop);
            source2->TraceConnectWithoutContext(
                "Tx", MakeBoundCallback(&setIdTag, 2, i_app + n_apps));
            sender->AddApplication(source2);
        }
        if (rate_w3 > 0)
        {
            Ptr<CdfApplication> source3 = CreateObjectWithAttributes<CdfApplication>(
                "Remote", recvAddr, "Protocol", TCP,
                "DataRate", DataRateValue(rate_w3), "CdfFile", StringValue(w3),
                "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
                "StopTime", simStop);
            source3->TraceConnectWithoutContext(
                "Tx", MakeBoundCallback(&setIdTag, 3, i_app + (2 * n_apps)));
            sender->AddApplication(source3);
        }
    }

    if (congestion > 0)
    {
        NS_LOG_INFO("Configure congestion app.");
        // Just blast UDP traffic from time to time
        Ptr<Application> congestion_sink = CreateObjectWithAttributes<PacketSink>(
            "Local", AddressValue(InetSocketAddress(addrReceiver, 2100)),
            "Protocol", UDP, "StartTime", simStart, "StopTime", simStop);
        receiver->AddApplication(congestion_sink);

        Ptr<Application> congestion_source = CreateObjectWithAttributes<OnOffApplication>(
            "Remote", AddressValue(InetSocketAddress(addrReceiver, 2100)),
            "Protocol", UDP,
            "OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"),
            "OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"),
            "DataRate", DataRateValue(congestion),
            "StartTime", TimeValue(Seconds(trafficStart->GetValue())),
            "StopTime", simStop);
        disturbance->AddApplication(congestion_source);
    }
    else
    {
        NS_LOG_INFO("No congestion.");
    }
    NS_LOG_INFO("Install Tracing");
    AsciiTraceHelper asciiTraceHelper;

    // Log (one-way) delay from sender to receiver (excludes other sources).
    std::stringstream trackfilename;
    // trackfilename << prefix << "_delays.csv";
    trackfilename << prefix << ".csv"; // only one file for now.
    auto trackfile = asciiTraceHelper.CreateFileStream(trackfilename.str());
    sender->GetDevice(0)->TraceConnectWithoutContext(
        "MacTx", MakeCallback(&setTimeTag));
    receiver->GetDevice(0)->TraceConnectWithoutContext(
        "MacRx", MakeBoundCallback(&logPacketInfo, trackfile));

    //csma.EnablePcapAll("csma-bridge", false);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    
    //
    // Now, do the actual simulation.
    //
    NS_LOG_INFO("Run Simulation.");
    Simulator::Stop(stopTime);
    Simulator::Run();
    Simulator::Destroy();
    NS_LOG_INFO("Done.");

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "Elapsed time: " << elapsed_seconds.count() << "s\n";
}
