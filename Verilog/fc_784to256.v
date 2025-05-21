module fc_784to256 (
    input wire clk,
    input wire rst_n,
    input wire i_valid,
    input wire [783:0] i_data,
    input wire [783:0] i_weight,
    input wire [9:0] i_threshold,
    output reg o_result,
    output reg o_valid
);

    integer i;

    // Stage 0
    reg [783:0] r_data_s0, r_weight_s0;
    reg [9:0]   r_threshold_s0;
    reg         valid_s0;

    // Stage 1
    reg [783:0] r_xnor_s1;
    reg         valid_s1;

    // Add Tree
    reg valid_sum1;
    reg valid_sum2;
    reg valid_sum3;
    reg valid_sum4;
    reg valid_sum5;
    reg valid_sum6;
    reg valid_sum7;
    reg valid_sum8;
    reg valid_sum9;
    reg valid_result;

    reg [1:0] sum1 [0:391];
    reg [2:0] sum2 [0:195];
    reg [3:0] sum3 [0:97];
    reg [4:0] sum4 [0:48];
    reg [5:0] sum5 [0:24];
    reg [6:0] sum6 [0:12];
    reg [7:0] sum7 [0:6];
    reg [8:0] sum8 [0:3];
    reg [9:0] sum9 [0:1];
    reg [10:0] result;

    // Stage 2
    reg [9:0]   r_popcount_s2;
    reg [9:0]   r_threshold_s2;
    reg         valid_s2;

    // ------------------------
    // Stage 0: Capture input
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_data_s0      <= 0;
            r_weight_s0    <= 0;
            r_threshold_s0 <= 0;
            valid_s0       <= 0;
        end else begin
            if (i_valid) begin
                r_data_s0      <= i_data;
                r_weight_s0    <= i_weight;
                r_threshold_s0 <= i_threshold;
                valid_s0       <= 1;
            end else begin
                valid_s0 <= 0;
            end
        end
    end

    // ------------------------
    // Stage 1: XNOR
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_xnor_s1 <= 0;
            valid_s1  <= 0;
        end else begin
            if (valid_s0) begin
                r_xnor_s1 <= ~(r_data_s0 ^ r_weight_s0);
                valid_s1  <= 1;
            end else begin
                valid_s1 <= 0;
            end
        end
    end

    // ------------------------
    // Stage 2: Popcount + Threshold
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_sum1 <= 0;
            valid_sum2 <= 0;            
            valid_sum3 <= 0;
            valid_sum4 <= 0;
            valid_sum5 <= 0;
            valid_sum6 <= 0;
            valid_sum7 <= 0;
            valid_sum8 <= 0;
            valid_sum9 <= 0;
            valid_result <= 0;            
        end else begin
            valid_sum1  <= valid_s1;
            valid_sum2  <= valid_sum1 ;           
            valid_sum3  <= valid_sum2;
            valid_sum4  <= valid_sum3;
            valid_sum5  <= valid_sum4;
            valid_sum6  <= valid_sum5;
            valid_sum7  <= valid_sum6;
            valid_sum8  <= valid_sum7;
            valid_sum9  <= valid_sum8;
            valid_result <= valid_sum9;           
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 392; i = i + 1)
                sum1[i] <= 0;
            for (i = 0; i < 196; i = i + 1)
                sum2[i] <= 0;
            for (i = 0; i < 98; i = i + 1) 
                sum3[i] <= 0;
            for (i = 0; i < 49; i = i + 1)
                sum4[i] <= 0;
            for (i = 0; i < 25; i = i + 1)
                sum5[i] <= 0;
            for (i = 0; i < 13; i = i + 1)
                sum6[i] <= 0;
            for (i = 0; i < 7; i = i + 1)
                sum7[i] <= 0;
            for (i = 0; i < 3; i = i + 1)
                sum8[i] <= 0;
            sum9[0] <= 0;
            sum9[1] <= 0;
            result  <= 0;

        end else begin
            if (valid_sum1) begin
                for (i = 0; i < 392; i = i + 1) begin
                    sum1[i] <= r_xnor_s1[i] + r_xnor_s1[392+i];
                end
            end

            if (valid_sum2) begin
                for (i = 0; i < 196; i = i + 1) begin
                    sum2[i] <= sum1[i] + sum1[196+i];
                end
            end

            if (valid_sum3) begin
                for (i = 0; i < 98; i = i + 1) begin
                    sum3[i] <= sum2[i] + sum2[98+i];
                end
            end

            if (valid_sum4) begin
                for (i = 0; i < 49; i = i + 1) begin
                    sum4[i] <= sum3[i] + sum3[49+i];
                end
            end

            if (valid_sum5) begin
                for (i = 0; i < 24; i = i + 1) begin
                    sum5[i] <= sum4[i] + sum4[24+i];
                end
                sum5[24] <= sum4[48];
            end

            if (valid_sum6) begin
                for (i = 0; i < 12; i = i + 1) begin
                    sum6[i] <= sum5[i] + sum5[12+i];
                end
                sum6[12] <= sum5[24];
            end

            if (valid_sum7) begin
                for (i = 0; i < 6; i = i + 1) begin
                    sum7[i] <= sum6[i] + sum6[6+i];
                end
                sum7[6] <= sum6[12];
            end

            if (valid_sum8) begin
                for (i = 0; i < 3; i = i + 1) begin
                    sum8[i] <= sum7[i] + sum7[3+i];
                end
                sum8[3] <= sum7[6];
            end

            if (valid_sum9) begin
                sum9[0] <= sum8[0] + sum8[1];
                sum9[1] <= sum8[2] + sum8[3];
            end

            if (valid_result) begin
                result <= sum9[0] + sum9[1];
            end

        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_popcount_s2   <= 0;
            r_threshold_s2 <= 0;
            valid_s2     <= 0;
        end else begin
            if (valid_result) begin
                r_popcount_s2     <= result;
                r_threshold_s2    <= r_threshold_s0;  // from stage 0
                valid_s2          <= 1;
            end else begin
                valid_s2 <= 0;
            end
        end
    end

    // ------------------------
    // Stage 3: Final compare
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_result <= 0;
            o_valid  <= 0;
        end else begin
            if (valid_s2) begin
                o_result <= (r_popcount_s2 >= r_threshold_s2);
                o_valid  <= 1;
            end else begin
                o_valid <= 0;
            end
        end
    end
endmodule
